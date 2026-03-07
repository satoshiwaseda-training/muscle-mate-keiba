"""netkeiba + JRA + OpenMeteo scraper.

Public data sources:
  - race.netkeiba.com  : race list, shutuba (entries), results, training times
  - db.netkeiba.com    : horse profiles (recent form, bloodline, weight history),
                         jockey stats (win rate, G1 recovery), trainer stats
  - api.open-meteo.com : weather forecast/history (free, no API key)
  - www.jra.go.jp      : fallback race list
"""

import time
import re
import json
from datetime import date, datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.2  # polite crawl rate

# ── 競馬場 → GPS座標 (OpenMeteo用) ──────────────────────────
VENUE_COORDS = {
    "東京": (35.7402, 139.5003),
    "中山": (35.7775, 139.9182),
    "阪神": (34.8431, 135.3542),
    "京都": (34.9072, 135.7680),
    "中京": (35.1072, 136.9248),
    "小倉": (33.8506, 130.8592),
    "新潟": (37.8985, 139.0567),
    "函館": (41.7749, 140.6953),
    "札幌": (43.0538, 141.3295),
    "福島": (37.7508, 140.4755),
}

# ── 輸送距離テーブル (km概算) ─────────────────────────────
TRANSPORT_DISTANCE = {
    "栗東": {"中山": 550, "東京": 500, "新潟": 600, "函館": 1200, "札幌": 1150,
              "阪神": 20, "京都": 10, "中京": 100, "小倉": 280, "福島": 650},
    "美浦": {"阪神": 500, "京都": 480, "中京": 380, "小倉": 750,
              "中山": 30, "東京": 40, "新潟": 250, "函館": 800, "札幌": 850, "福島": 120},
}

# ── 調教師名 → 所属 (主要100名) ──────────────────────────
TRAINER_STABLE = {
    # 栗東
    "矢作芳人": "栗東", "池添学": "栗東", "音無秀孝": "栗東", "中竹和也": "栗東",
    "西村真幸": "栗東", "斉藤崇史": "栗東", "杉山晴紀": "栗東", "友道康夫": "栗東",
    "中内田充正": "栗東", "高野友和": "栗東", "藤岡健一": "栗東", "松下武士": "栗東",
    "安田翔伍": "栗東", "吉村圭司": "栗東", "須貝尚介": "栗東", "池江泰寿": "栗東",
    "橋田満": "栗東", "清水久詞": "栗東", "武幸四郎": "栗東", "宮本博": "栗東",
    "今野貞一": "栗東", "石坂正": "栗東", "牧田和弥": "栗東", "千田輝彦": "栗東",
    "佐々木晶三": "栗東", "角居勝彦": "栗東", "福永祐一": "栗東",
    # 美浦
    "国枝栄": "美浦", "藤沢和雄": "美浦", "戸田博文": "美浦", "尾関知人": "美浦",
    "堀宣行": "美浦", "手塚貴久": "美浦", "田中剛": "美浦", "木村哲也": "美浦",
    "高柳瑞樹": "美浦", "鈴木伸尋": "美浦", "小桧山悟": "美浦", "宮田敬介": "美浦",
    "萩原清": "美浦", "武井亮": "美浦", "水野貴広": "美浦", "上原博之": "美浦",
    "和田正一郎": "美浦", "伊藤大士": "美浦", "青木孝文": "美浦", "牧浦充徳": "美浦",
    "菊沢隆徳": "美浦", "林徹": "美浦", "奥村武": "美浦", "南美之": "美浦",
}

# ── パドックコメント NLP スコアリング辞書 ─────────────────
PADDOCK_NLP = {
    "vascularity_index": {
        # 皮膚の薄さ・血管の見え方 → 仕上げの完成度
        "positive": ["血管", "皮膚が薄", "張り艶", "毛艶", "光沢", "ツヤ", "輝",
                     "ピカピカ", "締まり", "シャープ", "引き締まり", "仕上がり"],
        "negative": ["くすんで", "ぼてっと", "太め", "太目", "毛ヅヤ悪",
                     "艶がない", "ぼんやり", "疲労感"],
    },
    "hindquarter_power": {
        # トモのパンプアップ → 推進力ポテンシャル
        "positive": ["トモ", "後肢", "後脚", "推進力", "パンプ", "筋肉",
                     "発達", "しっかり", "力強", "たくましい", "充実"],
        "negative": ["トモが甘", "後肢が細", "推進力不足", "腰が甘", "ヒップが小"],
    },
    "gait_fluidity": {
        # 歩様の滑らかさ・首の使い方 → 重心移動の効率
        "positive": ["踏み込み", "首の使", "スムーズ", "柔軟", "リズム",
                     "軽快", "軽やか", "伸び伸び", "歩様良", "フットワーク", "しなやか"],
        "negative": ["硬い", "こわばり", "ガタガタ", "ぎこちない",
                     "歩様が固", "短い歩幅", "首が固"],
    },
}

# ── 調教コメント NLP スコアリング辞書 ──────────────────────
TRAINING_NLP = {
    "coat_gloss": {
        "positive": ["毛艶良", "毛並み良", "光沢あり", "ツヤあり", "毛色輝", "仕上がり上々"],
        "negative": ["毛艶悪", "毛並み悪", "くすんだ", "毛色が悪", "疲労感"],
    },
    "stride_quality": {
        "positive": ["踏み込み深", "大きなフォーム", "伸びのあるストライド",
                     "力強い踏み込み", "蹴り込み鋭", "ピッチ上昇"],
        "negative": ["踏み込み浅", "小さいフォーム", "短い歩幅", "こじんまり"],
    },
    "weight_status": {
        # negative = 太め残り フラグ
        "positive": ["絞れて", "仕上がり", "ベスト体型", "軽快", "シャープ", "スッキリ"],
        "negative": ["太め残", "太め", "重め", "仕上がり途上", "まだ太", "過剰"],
    },
}

# ── 馬主 → 外厩推定 ───────────────────────────────────────
RITTO_MAP = {
    "サンデーレーシング": "NF天栄", "キャロットファーム": "NF天栄",
    "シルクレーシング": "NF天栄", "吉田和美": "NF天栄", "吉田勝己": "NFしがらき",
    "金子真人": "NFしがらき", "社台レースホース": "社台外厩",
    "ゴドルフィン": "海外調整", "ダノックス": "乗馬クラブ系",
}


# ═══════════════════════════════════════════════════════════
# NLP Bio-Mechanical Analysis Helpers
# ═══════════════════════════════════════════════════════════

def _score_text(text: str, dimension_dict: dict) -> dict:
    """
    Score a text string against a keyword dictionary.
    Returns {dimension: float} where each value is in [-1.0, 1.0].
    0.0 = neutral/no signal.
    """
    if not text:
        return {dim: 0.0 for dim in dimension_dict}
    scores = {}
    for dim, keywords in dimension_dict.items():
        pos_hits = sum(1 for kw in keywords.get("positive", []) if kw in text)
        neg_hits = sum(1 for kw in keywords.get("negative", []) if kw in text)
        total = pos_hits + neg_hits
        if total == 0:
            scores[dim] = 0.0
        else:
            scores[dim] = round((pos_hits - neg_hits) / total, 3)
    return scores


def parse_paddock_comment(comment: str) -> dict:
    """
    Convert a paddock comment string into three scientific proxy variables.

    Returns:
      vascularity_index:  float  (-1 to 1) — skin tightness / coat finish quality
      hindquarter_power:  float  (-1 to 1) — muscle pump / propulsion potential
      gait_fluidity:      float  (-1 to 1) — stride smoothness / centre-of-mass efficiency
    """
    return _score_text(comment, PADDOCK_NLP)


def parse_training_comment(comment: str) -> dict:
    """
    Convert a training evaluation string into bio-mechanical scores.

    Returns:
      coat_gloss:     float  (-1 to 1)
      stride_quality: float  (-1 to 1)
      weight_status:  float  (-1 to 1, negative = 太め残り)
    """
    return _score_text(comment, TRAINING_NLP)


def analyze_training_physics(lap_str: str) -> dict:
    """
    Derive exercise-physiology metrics from a lap time string.

    lap_str examples:
      "12.5-11.8-11.2"            (split times)
      "38.5-12.2-11.0"            (cumulative prefix + splits)
      "CW良馬場 追い切り 12.3-11.5-11.1" (mixed)

    Returns:
      final_split:       float  — ラスト1F(秒)
      acceleration_rate: float  — (前F - 最終F) / 前F; 正値 = 加速
      cardio_index:      float  — ラスト3F→2F→1F の加速の鋭さ (心肺機能代理指標)
    """
    if not lap_str:
        return {"final_split": 0.0, "acceleration_rate": 0.0, "cardio_index": 0.0}

    splits = [float(s) for s in re.findall(r"\d+\.\d+", lap_str)]
    if not splits:
        return {"final_split": 0.0, "acceleration_rate": 0.0, "cardio_index": 0.0}
    if len(splits) == 1:
        return {"final_split": splits[0], "acceleration_rate": 0.0, "cardio_index": 0.0}

    last = splits[-1]
    second_last = splits[-2]
    acc_rate = round((second_last - last) / second_last, 4) if second_last > 0 else 0.0

    if len(splits) >= 3:
        third_last = splits[-3]
        stage1 = (third_last - second_last) / third_last if third_last > 0 else 0.0
        stage2 = (second_last - last) / second_last if second_last > 0 else 0.0
        cardio = round((stage1 + stage2) / 2, 4)
    else:
        cardio = acc_rate

    return {
        "final_split": last,
        "acceleration_rate": acc_rate,
        "cardio_index": cardio,
    }


def compute_best_weight_analysis(horse_id: str, current_weight: int,
                                  days_since_last_race: int = 0) -> dict:
    """
    Compare current race weight against historical best-performance weight.

    Reads stored 1st/2nd-place weight records from data_store.
    Returns:
      best_weight:     int | None
      deviation:       int   (+kg heavier, -kg lighter than best)
      classification:  str   ("ベスト体重帯" | "成長（バルクアップ）" |
                               "消耗/疲労による体重増" | "絞れた（仕上がり良好）" |
                               "消耗（体重減少）")
      confidence:      float  0–1 based on sample size
      sample_size:     int
    """
    from data_store import get_horse_profile

    profile = get_horse_profile(horse_id) or {}
    records = [r for r in profile.get("best_weight_records", []) if r.get("weight_kg")]
    if not records:
        return {"best_weight": None, "deviation": 0,
                "classification": "履歴なし", "confidence": 0.0, "sample_size": 0}

    best_weight = round(sum(r["weight_kg"] for r in records) / len(records))
    deviation = current_weight - best_weight

    if abs(deviation) <= 2:
        classification = "ベスト体重帯"
    elif deviation > 0:
        classification = "成長（バルクアップ）" if days_since_last_race >= 60 else "消耗/疲労による体重増"
    else:
        classification = "絞れた（仕上がり良好）" if days_since_last_race >= 30 else "消耗（体重減少）"

    return {
        "best_weight": best_weight,
        "deviation": deviation,
        "classification": classification,
        "confidence": round(min(1.0, len(records) / 5.0), 2),
        "sample_size": len(records),
    }


def build_transport_weight_profile(horse_id: str) -> dict:
    """
    Identify individual horse weaknesses by correlating historical
    transport distance, weather temperature, and finishing rank.

    Returns:
      patterns:                   list[str]  — human-readable weakness descriptions
      hot_transport_sensitivity:  bool
      long_transport_weakness:    bool
    """
    from data_store import get_horse_profile

    profile = get_horse_profile(horse_id) or {}
    log = profile.get("transport_weight_log", [])

    if len(log) < 3:
        return {"patterns": ["データ不足（3戦以上で分析開始）"],
                "hot_transport_sensitivity": False, "long_transport_weakness": False}

    patterns = []

    long = [e for e in log if e.get("transport_km", 0) >= 300]
    short = [e for e in log if e.get("transport_km", 0) < 300]
    long_transport_weakness = False
    if len(long) >= 2 and len(short) >= 2:
        lt_avg = sum(e.get("rank", 10) for e in long) / len(long)
        st_avg = sum(e.get("rank", 10) for e in short) / len(short)
        if lt_avg > st_avg + 1.5:
            long_transport_weakness = True
            patterns.append(
                f"長距離輸送(300km+)で平均着順が{lt_avg:.1f}着に低下 "
                f"（近距離平均{st_avg:.1f}着 vs 長距離平均{lt_avg:.1f}着）"
            )

    hot = [e for e in log if _is_hot_temp(e.get("weather_temp", ""))]
    hot_transport_sensitivity = False
    if len(hot) >= 2:
        hot_avg = sum(e.get("rank", 10) for e in hot) / len(hot)
        if hot_avg > 4.0:
            hot_transport_sensitivity = True
            patterns.append(
                f"高温時(25℃+)のレースで着順低下傾向（平均{hot_avg:.1f}着）"
            )

    # Weight-change patterns
    weight_diffs = []
    sorted_log = sorted(log, key=lambda x: x.get("date", ""))
    for i in range(1, len(sorted_log)):
        prev_w = sorted_log[i - 1].get("weight_kg", 0)
        curr_w = sorted_log[i].get("weight_kg", 0)
        if prev_w and curr_w:
            weight_diffs.append(curr_w - prev_w)
    if weight_diffs:
        avg_delta = sum(weight_diffs) / len(weight_diffs)
        if avg_delta > 3:
            patterns.append(f"レース間の平均体重増加傾向 (+{avg_delta:.1f}kg/戦)")
        elif avg_delta < -3:
            patterns.append(f"レース間の平均体重減少傾向 ({avg_delta:.1f}kg/戦) — 消耗注意")

    if not patterns:
        patterns.append("顕著な個体弱点パターン未検出")

    return {
        "patterns": patterns,
        "hot_transport_sensitivity": hot_transport_sensitivity,
        "long_transport_weakness": long_transport_weakness,
    }


def _is_hot_temp(weather_temp: str) -> bool:
    """Return True if the temperature string indicates >= 25°C."""
    m = re.search(r"(\d+(?:\.\d+)?)", str(weather_temp))
    if m:
        try:
            return float(m.group(1)) >= 25.0
        except Exception:
            pass
    return False


# ═══════════════════════════════════════════════════════════
# HTTP helper
# ═══════════════════════════════════════════════════════════

def _get(url: str, params: dict = None, encoding: str = None,
         delay: float = REQUEST_DELAY) -> Optional[BeautifulSoup]:
    try:
        time.sleep(delay)
        resp = requests.get(url, headers=HEADERS, params=params, timeout=12)
        resp.raise_for_status()
        if encoding:
            resp.encoding = encoding
        else:
            resp.encoding = resp.apparent_encoding
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"[scraper] GET {url} failed: {e}")
        return None


def _get_json(url: str, params: dict = None) -> Optional[dict]:
    """Fetch JSON endpoint (used for OpenMeteo)."""
    try:
        time.sleep(0.3)
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[scraper] JSON GET {url} failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# 1. Weather — OpenMeteo (free, no API key)
# ═══════════════════════════════════════════════════════════

def fetch_weather(venue: str, race_date: date, race_hour: int = 15) -> dict:
    """
    Fetch weather for a venue on race day from OpenMeteo.
    Returns {temperature, precipitation, windspeed, weather_code, description}.
    """
    coords = None
    for name, coord in VENUE_COORDS.items():
        if name in venue:
            coords = coord
            break
    if coords is None:
        return {"temperature": "不明", "precipitation": "不明",
                "windspeed": "不明", "description": "座標不明"}

    lat, lon = coords
    today = date.today()

    if race_date >= today:
        # Forecast
        data = _get_json(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,precipitation,windspeed_10m,weathercode",
                "timezone": "Asia/Tokyo",
                "start_date": race_date.isoformat(),
                "end_date": race_date.isoformat(),
            },
        )
    else:
        # Historical archive
        data = _get_json(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,precipitation,windspeed_10m,weathercode",
                "timezone": "Asia/Tokyo",
                "start_date": race_date.isoformat(),
                "end_date": race_date.isoformat(),
            },
        )

    if not data or "hourly" not in data:
        return {"temperature": "取得失敗", "precipitation": "取得失敗",
                "windspeed": "取得失敗", "description": "APIエラー"}

    hourly = data["hourly"]
    times = hourly.get("time", [])
    target = f"{race_date.isoformat()}T{race_hour:02d}:00"
    idx = times.index(target) if target in times else min(race_hour, len(times) - 1)

    temp = hourly["temperature_2m"][idx] if hourly.get("temperature_2m") else "?"
    precip = hourly["precipitation"][idx] if hourly.get("precipitation") else "?"
    wind = hourly["windspeed_10m"][idx] if hourly.get("windspeed_10m") else "?"
    wcode = hourly["weathercode"][idx] if hourly.get("weathercode") else 0

    desc = _weather_code_to_ja(wcode)
    return {
        "temperature": f"{temp}℃",
        "precipitation": f"{precip}mm",
        "windspeed": f"{wind}km/h",
        "description": desc,
        "source": "OpenMeteo",
    }


def _weather_code_to_ja(code: int) -> str:
    if code == 0:
        return "快晴"
    elif code in (1, 2, 3):
        return "晴〜曇"
    elif code in range(45, 57):
        return "霧"
    elif code in range(61, 68):
        return "雨"
    elif code in range(71, 78):
        return "雪"
    elif code in range(80, 83):
        return "にわか雨"
    elif code in range(95, 100):
        return "雷雨"
    return "不明"


# ═══════════════════════════════════════════════════════════
# 2. Horse detail — db.netkeiba.com
# ═══════════════════════════════════════════════════════════

def fetch_horse_detail(horse_id: str) -> dict:
    """
    Fetch horse profile from db.netkeiba.com/horse/{horse_id}/.
    Returns recent 5 races, bloodline (sire/dam), body weight trend.
    """
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    soup = _get(url)
    if soup is None:
        return {}

    result = {"horse_id": horse_id, "recent_races": [], "sire": "", "dam": "", "weight_trend": []}

    # Bloodline
    for row in soup.select("table.db_prof_table tr, .horse_title tr"):
        cells = row.find_all("td")
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            val = cells[1].get_text(strip=True)
            if "父" == label:
                result["sire"] = val
            elif "母" == label:
                result["dam"] = val

    # Recent race results table (直近成績)
    for tbl in soup.select("table.race_table_01, table.db_h_race_results"):
        rows = tbl.select("tr")[1:6]  # up to 5 races
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 10:
                continue
            try:
                result["recent_races"].append({
                    "date": cells[0].get_text(strip=True),
                    "venue": cells[1].get_text(strip=True),
                    "race_name": cells[4].get_text(strip=True),
                    "rank": cells[11].get_text(strip=True) if len(cells) > 11 else cells[-3].get_text(strip=True),
                    "time": cells[17].get_text(strip=True) if len(cells) > 17 else "",
                    "weight": cells[20].get_text(strip=True) if len(cells) > 20 else "",
                    "jockey": cells[12].get_text(strip=True) if len(cells) > 12 else "",
                })
            except Exception:
                continue
        if result["recent_races"]:
            break

    # Body weight trend (last 5 recorded weights)
    weights = re.findall(r"(\d{3,4})\(([+-]?\d+)\)", soup.get_text())
    result["weight_trend"] = [f"{w}kg({d})" for w, d in weights[:5]]

    return result


# ═══════════════════════════════════════════════════════════
# 3. Jockey stats — db.netkeiba.com
# ═══════════════════════════════════════════════════════════

def fetch_jockey_stats(jockey_id: str) -> dict:
    """
    Fetch jockey career stats from db.netkeiba.com/jockey/{jockey_id}/.
    Returns win_rate, place_rate, g1_wins, single_recovery_rate.
    """
    url = f"https://db.netkeiba.com/jockey/result/recent/{jockey_id}/"
    soup = _get(url)
    if soup is None:
        return {}

    stats = {"jockey_id": jockey_id, "win_rate": "", "place_rate": "",
             "g1_wins": "", "single_recovery": ""}

    tbl = soup.select_one("table.db_prof_table, table.race_table_01")
    if tbl:
        rows = tbl.select("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 6:
                try:
                    rides = int(cells[0].get_text(strip=True).replace(",", ""))
                    wins = int(cells[1].get_text(strip=True).replace(",", ""))
                    seconds = int(cells[2].get_text(strip=True).replace(",", ""))
                    thirds = int(cells[3].get_text(strip=True).replace(",", ""))
                    if rides > 0:
                        stats["win_rate"] = f"{wins/rides:.1%}"
                        stats["place_rate"] = f"{(wins+seconds+thirds)/rides:.1%}"
                except Exception:
                    pass
                break

    # G1 wins: count from results filtered by grade
    grade_url = f"https://db.netkeiba.com/jockey/result/recent/{jockey_id}/?grade=G1"
    g1_soup = _get(grade_url)
    if g1_soup:
        g1_wins_match = re.search(r"(\d+)勝", g1_soup.get_text())
        stats["g1_wins"] = g1_wins_match.group(1) + "勝" if g1_wins_match else "?"

    return stats


# ═══════════════════════════════════════════════════════════
# 4. Trainer stats — db.netkeiba.com
# ═══════════════════════════════════════════════════════════

def fetch_trainer_stats(trainer_id: str) -> dict:
    """Fetch trainer win rate and specialty distance."""
    url = f"https://db.netkeiba.com/trainer/result/recent/{trainer_id}/"
    soup = _get(url)
    if soup is None:
        return {}

    stats = {"trainer_id": trainer_id, "win_rate": "", "place_rate": ""}
    tbl = soup.select_one("table.db_prof_table, table.race_table_01")
    if tbl:
        for row in tbl.select("tr"):
            cells = row.find_all("td")
            if len(cells) >= 4:
                try:
                    rides = int(cells[0].get_text(strip=True).replace(",", ""))
                    wins = int(cells[1].get_text(strip=True).replace(",", ""))
                    place = int(cells[2].get_text(strip=True).replace(",", "")) + int(cells[3].get_text(strip=True).replace(",", ""))
                    if rides > 0:
                        stats["win_rate"] = f"{wins/rides:.1%}"
                        stats["place_rate"] = f"{(wins+place)/rides:.1%}"
                except Exception:
                    pass
                break
    return stats


# ═══════════════════════════════════════════════════════════
# 5. Training times — race.netkeiba.com/oikiri
# ═══════════════════════════════════════════════════════════

def fetch_training_times(race_id: str) -> list[dict]:
    """
    Fetch training (追い切り) times for all horses in a race.
    Returns list of {name, course, lap, evaluation}.
    """
    url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
    soup = _get(url)
    if soup is None:
        return []

    results = []
    for row in soup.select("tr.OikiriList_Row, tr[class*='HorseList']"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        try:
            name_tag = row.select_one(".HorseName, .Horse_Name")
            name = name_tag.get_text(strip=True) if name_tag else cells[1].get_text(strip=True)
            course_tag = row.select_one(".Course, .TrackCourse")
            course = course_tag.get_text(strip=True) if course_tag else cells[2].get_text(strip=True)
            time_tag = row.select_one(".Time, .OikiriTime")
            lap = time_tag.get_text(strip=True) if time_tag else cells[3].get_text(strip=True)
            eval_tag = row.select_one(".Hyouka, .Evaluation")
            evaluation = eval_tag.get_text(strip=True) if eval_tag else ""
            results.append({"name": name, "course": course, "lap": lap, "evaluation": evaluation})
        except Exception:
            continue

    return results


# ═══════════════════════════════════════════════════════════
# 6. Race list — netkeiba
# ═══════════════════════════════════════════════════════════

def get_this_week_race_dates() -> tuple["date", "date"]:
    """
    今週の開催日（土曜・日曜）を返す。
    木曜以降なら「今週末」、月〜水なら「直近の土日」を返す。
    """
    today = date.today()
    weekday = today.weekday()  # 月=0 … 日=6
    # days_to_saturday: 0=月曜→5日後, 5=土曜→0日後, 6=日曜→6日後
    days_to_sat = (5 - weekday) % 7
    saturday = today + __import__("datetime").timedelta(days=days_to_sat)
    sunday = saturday + __import__("datetime").timedelta(days=1)
    return saturday, sunday


def fetch_race_list_netkeiba(race_date: date) -> list[dict]:
    """
    race_list_sub.html (AJAXエンドポイント) から G1/G2/G3 を取得。
    Icon_GradeType1=G1, Icon_GradeType2=G2, Icon_GradeType3=G3
    """
    date_str = race_date.strftime("%Y%m%d")
    soup = _get(
        "https://race.netkeiba.com/top/race_list_sub.html",
        params={"kaisai_date": date_str},
    )
    if soup is None:
        return []

    # 会場名を先に収集: .RaceList_DataTitle の直後のリストが同会場
    # 構造: RaceList_Box > RaceList_DataTitle + RaceList_DataList > li.RaceList_DataItem
    venue_map: dict[str, str] = {}
    for box in soup.select(".RaceList_Box"):
        venue_tag = box.select_one(".RaceList_DataTitle")
        venue_text = venue_tag.get_text(strip=True) if venue_tag else ""
        # 「2回中山4日目」→ 「中山」を抽出
        venue_short = re.sub(r"\d+回|\d+日目", "", venue_text).strip()
        for li in box.select("li.RaceList_DataItem"):
            a = li.select_one("a[href*='race_id']")
            if not a:
                continue
            m = re.search(r"race_id=(\d+)", a.get("href", ""))
            if m:
                venue_map[m.group(1)] = venue_short

    races = []
    for li in soup.select("li.RaceList_DataItem"):
        # ── グレード判定 (Type1=G1, Type2=G2, Type3=G3) ────
        grade = ""
        for tag in li.select(".Icon_GradeType"):
            for cls in tag.get("class", []):
                if cls == "Icon_GradeType1":
                    grade = "G1"
                elif cls == "Icon_GradeType2":
                    grade = "G2"
                elif cls == "Icon_GradeType3":
                    grade = "G3"
        if not grade:
            continue

        # ── race_id ─────────────────────────────────────────
        a = li.select_one("a[href*='race_id']")
        if not a:
            continue
        m = re.search(r"race_id=(\d+)", a.get("href", ""))
        if not m:
            continue
        race_id = m.group(1)

        # ── レース名 ────────────────────────────────────────
        name_tag = li.select_one(".ItemTitle")
        race_name = name_tag.get_text(strip=True) if name_tag else "不明"

        # ── 発走時刻 ────────────────────────────────────────
        time_tag = li.select_one(".RaceList_Itemtime")
        race_time = time_tag.get_text(strip=True) if time_tag else ""

        # ── 会場 ────────────────────────────────────────────
        venue = venue_map.get(race_id, "")

        races.append({
            "race_id": race_id,
            "race_name": f"{race_name} ({grade})",
            "grade": grade,
            "venue": venue,
            "time": race_time,
            "weather": "",
            "track_condition": "",
            "race_date": race_date.isoformat(),
            "url": f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}",
            "source": "netkeiba",
        })

    return races


# ═══════════════════════════════════════════════════════════
# 7. Race metadata — netkeiba shutuba
# ═══════════════════════════════════════════════════════════

def fetch_race_info_netkeiba(race_id: str) -> dict:
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    soup = _get(url)
    info = {"weather": "", "temperature": "", "cushion_value": "",
            "track_condition": "", "distance": "", "surface": ""}
    if soup is None:
        return info

    data_tag = soup.select_one(".RaceData01") or soup.select_one(".RaceHeader_Data")
    if data_tag:
        text = data_tag.get_text(" ", strip=True)
        for pattern, key in [
            (r"天候[:：]\s*(\S+)", "weather"),
            (r"気温[:：]\s*([\d.]+)", "temperature"),
            (r"馬場[:：]\s*(\S+)", "track_condition"),
            (r"クッション値[:：]\s*([\d.]+)", "cushion_value"),
            (r"(芝|ダート)([\d,]+)m", "surface"),
        ]:
            m = re.search(pattern, text)
            if m:
                info[key] = m.group(0) if key == "surface" else m.group(1)
    return info


# ═══════════════════════════════════════════════════════════
# 8. Entries — netkeiba shutuba (with horse/jockey IDs)
# ═══════════════════════════════════════════════════════════

def fetch_entries_netkeiba(race_id: str, venue: str = "") -> list[dict]:
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    soup = _get(url)
    if soup is None:
        return _mock_entries(race_id)

    horses = []
    for row in soup.select("tr.HorseList"):
        cells = row.find_all("td")
        if len(cells) < 8:
            continue
        try:
            num = cells[1].get_text(strip=True)

            name_tag = row.select_one(".HorseName a") or row.select_one("td:nth-child(4) a")
            name = name_tag.get_text(strip=True) if name_tag else ""
            # Extract horse ID from href
            horse_id = ""
            if name_tag:
                href = name_tag.get("href", "")
                m = re.search(r"/horse/(\d+)", href)
                horse_id = m.group(1) if m else ""

            jockey_tag = row.select_one(".Jockey a")
            jockey = jockey_tag.get_text(strip=True) if jockey_tag else ""
            jockey_id = ""
            if jockey_tag:
                href = jockey_tag.get("href", "")
                m = re.search(r"/jockey/(?:result/recent/)?(\w+)", href)
                jockey_id = m.group(1) if m else ""

            trainer_tag = row.select_one(".Trainer a")
            trainer = trainer_tag.get_text(strip=True) if trainer_tag else ""
            trainer_id = ""
            if trainer_tag:
                href = trainer_tag.get("href", "")
                m = re.search(r"/trainer/(?:result/recent/)?(\w+)", href)
                trainer_id = m.group(1) if m else ""

            owner_tag = row.select_one(".Owner a")
            owner = owner_tag.get_text(strip=True) if owner_tag else ""

            weight_tag = row.select_one(".Txt_C")
            weight = weight_tag.get_text(strip=True) if weight_tag else ""

            odds_tag = row.select_one(".Odds") or row.select_one(".Popular")
            odds = odds_tag.get_text(strip=True) if odds_tag else "0"

            # Age / sex
            age_tag = row.select_one(".Age") or (cells[4] if len(cells) > 4 else None)
            age = age_tag.get_text(strip=True) if age_tag else ""

            stable = TRAINER_STABLE.get(trainer, _guess_stable(trainer))
            ritto = _estimate_ritto(owner)
            transport = _transport_stress(stable, venue)

            horses.append({
                "number": num,
                "name": name,
                "horse_id": horse_id,
                "jockey": jockey,
                "jockey_id": jockey_id,
                "trainer": trainer,
                "trainer_id": trainer_id,
                "owner": owner,
                "age": age,
                "weight": weight,
                "odds": odds,
                "stable": stable,
                "ritto": ritto,
                "transport_stress": transport,
                # Filled by enrich_entries():
                "recent_form": "",
                "bloodline": "",
                "weight_trend": "",
                "jockey_win_rate": "",
                "jockey_g1_wins": "",
                "trainer_win_rate": "",
                "training_eval": "",
            })
        except Exception:
            continue

    return horses if horses else _mock_entries(race_id)


def enrich_entries(horses: list[dict], race_id: str,
                   progress_callback=None) -> list[dict]:
    """
    Fetch detailed public data for each horse:
      - Horse recent races + bloodline (db.netkeiba)
      - Jockey stats (db.netkeiba)
      - Training times (netkeiba oikiri)
    This is slow (~N*3 requests). Call separately from a "詳細取得" button.
    """
    training_map = {t["name"]: t for t in fetch_training_times(race_id)}

    for i, h in enumerate(horses):
        if progress_callback:
            progress_callback(i, len(horses), h["name"])

        # Horse detail
        if h.get("horse_id"):
            detail = fetch_horse_detail(h["horse_id"])
            if detail:
                races = detail.get("recent_races", [])
                h["recent_form"] = " → ".join(
                    f'{r["date"]} {r["race_name"]} {r["rank"]}着'
                    for r in races[:3]
                )
                h["bloodline"] = f"父:{detail.get('sire','?')} 母:{detail.get('dam','?')}"
                h["weight_trend"] = " ".join(detail.get("weight_trend", [])[:3])

        # Jockey stats
        if h.get("jockey_id"):
            jstats = fetch_jockey_stats(h["jockey_id"])
            h["jockey_win_rate"] = jstats.get("win_rate", "")
            h["jockey_g1_wins"] = jstats.get("g1_wins", "")

        # Training eval + physics analysis
        train = training_map.get(h["name"], {})
        if train:
            lap_str = train.get("lap", "")
            eval_str = train.get("evaluation", "")
            h["training_eval"] = f"{train.get('course','')} {lap_str} {eval_str}"
            # Physics analysis
            h["training_physics"] = analyze_training_physics(lap_str)
            # NLP score of evaluation text
            h["training_nlp"] = parse_training_comment(eval_str)
        else:
            h["training_physics"] = {"final_split": 0.0, "acceleration_rate": 0.0, "cardio_index": 0.0}
            h["training_nlp"] = {}

        # Paddock NLP scores (from pre-filled training_eval text or evaluation field)
        paddock_text = h.get("paddock_comment", "")
        h["paddock_scores"] = parse_paddock_comment(paddock_text)

    return horses


# ═══════════════════════════════════════════════════════════
# 9. Result — netkeiba
# ═══════════════════════════════════════════════════════════

def fetch_result_netkeiba(race_id: str) -> Optional[dict]:
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    soup = _get(url)
    if soup is None:
        return None

    finishing_order = []
    for row in soup.select("tr.HorseList"):
        cells = row.find_all("td")
        if len(cells) < 6:
            continue
        try:
            rank = cells[0].get_text(strip=True)
            name_tag = row.select_one(".HorseName a")
            name = name_tag.get_text(strip=True) if name_tag else cells[3].get_text(strip=True)
            time_tag = row.select_one(".RecordTime")
            finish_time = time_tag.get_text(strip=True) if time_tag else ""
            odds_tag = row.select_one(".Odds")
            final_odds = odds_tag.get_text(strip=True) if odds_tag else ""
            finishing_order.append({
                "rank": int(rank) if rank.isdigit() else 99,
                "name": name, "time": finish_time, "odds": final_odds,
            })
        except Exception:
            continue

    payouts = {}
    for row in soup.select("tr.Payout_Detail_Table_Row, .Payout_Detail tr"):
        cells = row.find_all("td")
        if len(cells) >= 2:
            bet_type = cells[0].get_text(strip=True)
            val = cells[-1].get_text(strip=True).replace(",", "").replace("円", "")
            try:
                payouts[bet_type] = int(val)
            except ValueError:
                pass

    return {"finishing_order": finishing_order, "payouts": payouts} if finishing_order else None


# ═══════════════════════════════════════════════════════════
# 10. JRA fallback
# ═══════════════════════════════════════════════════════════

def fetch_race_list_jra(race_date: date) -> list[dict]:
    soup = _get("https://www.jra.go.jp/keiba/thisweek/", encoding="utf-8")
    if soup is None:
        return []
    races = []
    for link in soup.select("a[href*='race']"):
        text = link.get_text(strip=True)
        gm = re.search(r"(G[123])", text)
        if not gm:
            continue
        href = link.get("href", "")
        race_id = re.sub(r"[^0-9]", "", href)[-16:] or href
        races.append({
            "race_id": race_id, "race_name": text, "grade": gm.group(1),
            "venue": "", "time": "", "weather": "", "track_condition": "",
            "url": href if href.startswith("http") else f"https://www.jra.go.jp{href}",
            "source": "jra",
        })
    return races


# ═══════════════════════════════════════════════════════════
# Unified public API
# ═══════════════════════════════════════════════════════════

def fetch_race_list(race_date: date) -> list[dict]:
    races = fetch_race_list_netkeiba(race_date)
    if not races:
        races = fetch_race_list_jra(race_date)
    return races if races else []


def fetch_this_week_races() -> list[dict]:
    """
    今週の土曜・日曜両日のG1/G2/G3レースをまとめて取得する。
    """
    saturday, sunday = get_this_week_race_dates()
    races = []
    for d in (saturday, sunday):
        day_races = fetch_race_list(d)
        races.extend(day_races)
    return races


def fetch_entries(race_id: str, venue: str = "") -> list[dict]:
    entries = fetch_entries_netkeiba(race_id, venue)
    return entries if entries else _mock_entries(race_id)


def fetch_race_info(race_id: str) -> dict:
    return fetch_race_info_netkeiba(race_id)


def fetch_result(race_id: str) -> Optional[dict]:
    return fetch_result_netkeiba(race_id)


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _guess_stable(trainer: str) -> str:
    """Fallback heuristic for unknown trainers."""
    KURI_HINTS = {"池", "音", "杉", "友", "中内", "高野", "安田", "須", "石坂", "牧田"}
    MIHO_HINTS = {"国", "藤", "戸田", "尾", "堀", "手塚", "田中", "木村", "高柳", "鈴"}
    if any(h in trainer for h in KURI_HINTS):
        return "栗東"
    if any(h in trainer for h in MIHO_HINTS):
        return "美浦"
    return "不明"


def _transport_stress(stable: str, venue: str) -> str:
    for loc, table in TRANSPORT_DISTANCE.items():
        if loc in (stable or ""):
            for v, km in table.items():
                if v in (venue or ""):
                    if km < 50:
                        return f"低({km}km)"
                    elif km < 300:
                        return f"中({km}km)"
                    else:
                        return f"高({km}km)"
    return "不明"


def _estimate_ritto(owner: str) -> str:
    for key, val in RITTO_MAP.items():
        if key in (owner or ""):
            return val
    return "不明"


# ═══════════════════════════════════════════════════════════
# Mock data
# ═══════════════════════════════════════════════════════════

def _mock_race_list(race_date: date) -> list[dict]:
    return [
        {"race_id": f"mock_{race_date.strftime('%Y%m%d')}_01",
         "race_name": "【モック】高松宮記念 (G1)", "grade": "G1",
         "venue": "中京 芝1200m", "time": "15:40",
         "weather": "晴", "track_condition": "良", "url": "", "source": "mock"},
        {"race_id": f"mock_{race_date.strftime('%Y%m%d')}_02",
         "race_name": "【モック】毎日杯 (G3)", "grade": "G3",
         "venue": "阪神 芝1800m", "time": "15:00",
         "weather": "曇", "track_condition": "稍重", "url": "", "source": "mock"},
    ]


def _mock_entries(race_id: str) -> list[dict]:
    data = [
        ("1","ドウデュース",      "horse001","武豊",      "jky001","友道康夫","tr001","57","3.5",
         "サンデーレーシング","栗東","NF天栄",   "1着-1着-2着"),
        ("2","イクイノックス",    "horse002","C.ルメール", "jky002","木村哲也","tr002","58","2.1",
         "シルクレーシング", "美浦","NF天栄",   "1着-1着-1着"),
        ("3","リバティアイランド","horse003","川田将雅",   "jky003","中内田充正","tr003","55","4.8",
         "キャロットファーム","栗東","NF天栄",  "1着-2着-1着"),
        ("4","タスティエーラ",    "horse004","松山弘平",   "jky004","堀宣行",  "tr004","57","12.0",
         "サンデーレーシング","美浦","NF天栄",  "3着-2着-1着"),
        ("5","ソールオリエンス",  "horse005","横山武史",   "jky005","手塚貴久","tr005","57","15.0",
         "金子真人H",        "美浦","NFしがらき","2着-3着-4着"),
        ("6","ダノンベルーガ",    "horse006","横山典弘",   "jky006","堀宣行",  "tr006","57","18.0",
         "ダノックス",       "美浦","不明",      "4着-1着-2着"),
        ("7","ジャスティンパレス","horse007","鮫島克駿",   "jky007","杉山晴紀","tr007","57","22.0",
         "大塚亮一",         "栗東","不明",      "5着-3着-1着"),
        ("8","スターズオンアース","horse008","戸崎圭太",   "jky008","高柳瑞樹","tr008","55","9.0",
         "吉田和美",         "美浦","NF天栄",   "2着-1着-3着"),
    ]
    return [
        {
            "number": num, "name": name,
            "horse_id": hid, "jockey": jockey, "jockey_id": jid,
            "trainer": trainer, "trainer_id": tid,
            "age": "牡4", "weight": weight, "odds": odds,
            "owner": owner, "stable": stable, "ritto": ritto,
            "recent_form": form,
            "transport_stress": _transport_stress(stable, "中京"),
            "bloodline": "父:ハーツクライ 母:ドウデュース", "weight_trend": "478kg(+2)",
            "jockey_win_rate": "18%", "jockey_g1_wins": "3勝",
            "trainer_win_rate": "15%", "training_eval": "",
        }
        for num, name, hid, jockey, jid, trainer, tid, weight, odds, owner, stable, ritto, form in data
    ]
