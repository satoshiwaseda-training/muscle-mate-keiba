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
import html
from datetime import date, datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup


def _clean_text(text: str) -> str:
    """HTMLエンティティ・競馬印記号・制御文字をクリーニングする。"""
    if not text:
        return ""
    # HTMLエンティティをデコード (&#10003; → ✓ など)
    text = html.unescape(text)
    # 競馬印記号（◎本命 ○対抗 ▲単穴 △連下 ☆穴 ×消 など）を除去
    text = re.sub(r'[◎○◯▲△☆★×✓✔✗✘]', '', text)
    # 連続ハイフン・ダッシュ整理
    text = re.sub(r'-{2,}', '', text)
    # 制御文字・不可視文字除去
    text = re.sub(r'[\x00-\x1f\x7f\u200b\u200c\u200d\ufeff]', '', text)
    # 余分な空白を整理
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
        # Use raw bytes so BeautifulSoup reads the meta charset tag (EUC-JP etc.) correctly.
        # Passing resp.text causes double-encoding issues on EUC-JP pages (netkeiba).
        if encoding:
            return BeautifulSoup(resp.content, "html.parser", from_encoding=encoding)
        return BeautifulSoup(resp.content, "html.parser")
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

    # OpenMeteo パラメータ (新旧両対応: wind_speed_10m / weather_code)
    _hourly_vars = "temperature_2m,precipitation,wind_speed_10m,weather_code"
    base_params = {
        "latitude": lat, "longitude": lon,
        "hourly": _hourly_vars,
        "timezone": "Asia/Tokyo",
        "start_date": race_date.isoformat(),
        "end_date": race_date.isoformat(),
    }

    if race_date >= today:
        data = _get_json("https://api.open-meteo.com/v1/forecast", params=base_params)
    else:
        data = _get_json("https://archive-api.open-meteo.com/v1/archive", params=base_params)

    if not data or "hourly" not in data:
        return {"temperature": "取得失敗", "precipitation": "取得失敗",
                "windspeed": "取得失敗", "description": "APIエラー"}

    hourly = data["hourly"]
    times = hourly.get("time", [])
    target = f"{race_date.isoformat()}T{race_hour:02d}:00"
    idx = times.index(target) if target in times else min(race_hour, len(times) - 1)

    temp = hourly["temperature_2m"][idx] if hourly.get("temperature_2m") else "?"
    precip = hourly["precipitation"][idx] if hourly.get("precipitation") else "?"
    # 新パラメータ名優先、旧名フォールバック
    wind = (hourly.get("wind_speed_10m") or hourly.get("windspeed_10m") or [None])[idx] or "?"
    wcode = (hourly.get("weather_code") or hourly.get("weathercode") or [0])[idx] or 0

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
        if len(cells) < 7:
            continue
        try:
            # 枠番[0] 馬番[1] チェック[2] 馬名[3] 性齢[4] 斤量[5] 騎手[6] 調教師[7] 馬体重[8] オッズ[9]
            waku = cells[0].get_text(strip=True)
            num  = cells[1].get_text(strip=True)

            # 馬名 + horse_id
            horse_info_td = cells[3]
            name_tag = horse_info_td.select_one("a")
            name = name_tag.get_text(strip=True) if name_tag else horse_info_td.get_text(strip=True)
            horse_id = ""
            if name_tag:
                m = re.search(r"/horse/(\d+)", name_tag.get("href", ""))
                horse_id = m.group(1) if m else ""

            # 性齢
            age = cells[4].get_text(strip=True) if len(cells) > 4 else ""

            # 斤量
            weight = cells[5].get_text(strip=True) if len(cells) > 5 else ""

            # 騎手
            jockey_td = cells[6] if len(cells) > 6 else None
            jockey_tag = jockey_td.select_one("a") if jockey_td else None
            jockey = jockey_tag.get_text(strip=True) if jockey_tag else (jockey_td.get_text(strip=True) if jockey_td else "")
            jockey_id = ""
            if jockey_tag:
                m = re.search(r"/jockey/(?:result/recent/)?(\w+)", jockey_tag.get("href", ""))
                jockey_id = m.group(1) if m else ""

            # 調教師
            trainer_td = cells[7] if len(cells) > 7 else None
            trainer_tag = trainer_td.select_one("a") if trainer_td else None
            trainer = trainer_tag.get_text(strip=True) if trainer_tag else (trainer_td.get_text(strip=True) if trainer_td else "")
            trainer_id = ""
            if trainer_tag:
                m = re.search(r"/trainer/(?:result/recent/)?(\w+)", trainer_tag.get("href", ""))
                trainer_id = m.group(1) if m else ""

            # 馬体重
            weight_td = cells[8] if len(cells) > 8 else None
            horse_weight = weight_td.get_text(strip=True) if weight_td else ""

            # オッズ（出走前は---の場合あり）
            odds_td = cells[9] if len(cells) > 9 else None
            odds = odds_td.get_text(strip=True) if odds_td else "0"
            odds = odds.replace("---.-", "0").replace("---", "0")

            # 馬主（存在しない場合は空）
            owner_tag = row.select_one(".Owner a")
            owner = owner_tag.get_text(strip=True) if owner_tag else ""

            stable = TRAINER_STABLE.get(trainer, _guess_stable(trainer))
            ritto = _estimate_ritto(owner)
            transport = _transport_stress(stable, venue)

            if not name:
                continue

            horses.append({
                "number": num,
                "waku": waku,
                "name": name,
                "horse_id": horse_id,
                "jockey": jockey,
                "jockey_id": jockey_id,
                "trainer": trainer,
                "trainer_id": trainer_id,
                "owner": owner,
                "age": age,
                "weight": weight,
                "horse_weight": horse_weight,
                "odds": odds,
                "stable": stable,
                "ritto": ritto,
                "transport_stress": transport,
                # Filled by enrich_entries() — safe defaults to avoid KeyError
                "recent_form": "",
                "bloodline": "",
                "weight_trend": "",
                "jockey_win_rate": "",
                "jockey_g1_wins": "",
                "trainer_win_rate": "",
                "training_eval": "",
                "training_physics": {"final_split": 0.0, "acceleration_rate": 0.0, "cardio_index": 0.0},
                "training_nlp": {},
                "paddock_scores": {},
                "best_weight_analysis": {},
                "transport_profile": {},
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
            rank_text = cells[0].get_text(strip=True)
            if not rank_text.isdigit():
                continue  # 除外・取消・中止などをスキップ

            # 馬名: Horse_Info セルの <a> タグ、なければセルテキスト
            horse_info = row.select_one("td.Horse_Info a") or row.select_one(".HorseName a")
            name = horse_info.get_text(strip=True) if horse_info else cells[3].get_text(strip=True)
            horse_id = ""
            if horse_info:
                m = re.search(r"/horse/(\d+)", horse_info.get("href", ""))
                horse_id = m.group(1) if m else ""

            # タイム: Time クラスの最初のセル
            time_cell = row.select_one("td.Time")
            finish_time = time_cell.get_text(strip=True) if time_cell else ""

            # オッズ: Txt_R クラスの Odds セル（人気順位ではなくオッズ数値側）
            odds_cells = row.select("td.Odds")
            final_odds = ""
            for oc in odds_cells:
                txt = oc.get_text(strip=True)
                if "." in txt:  # 小数点ありがオッズ値
                    final_odds = txt
                    break
            if not final_odds and odds_cells:
                final_odds = odds_cells[-1].get_text(strip=True)

            # 馬体重
            weight_cell = row.select_one("td.Weight")
            horse_weight = weight_cell.get_text(strip=True) if weight_cell else ""

            # 騎手
            jockey_cell = row.select_one("td.Jockey a") or row.select_one("td.Jockey")
            jockey = jockey_cell.get_text(strip=True) if jockey_cell else ""

            finishing_order.append({
                "rank": int(rank_text),
                "name": name,
                "horse_id": horse_id,
                "jockey": jockey,
                "time": finish_time,
                "odds": final_odds,
                "horse_weight": horse_weight,
            })
        except Exception:
            continue

    # 払い戻し: ResultPaybackLeftWrap / ResultPaybackRightWrap
    payouts = _parse_netkeiba_payouts(soup)

    return {"finishing_order": finishing_order, "payouts": payouts} if finishing_order else None


def _parse_netkeiba_payouts(soup: BeautifulSoup) -> dict:
    """
    払い戻しセクションから単勝・複勝・馬連などを抽出する。
    各 <li> または行テキストが「馬券種 馬番 金額円 人気」の形式になっている。
    """
    payouts = {}
    bet_types = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]
    for wrap in soup.select("div.ResultPaybackLeftWrap, div.ResultPaybackRightWrap, div.Payout"):
        for item in wrap.select("tr"):  # 払い戻しはtr要素に入っている
            text = item.get_text(" ", strip=True)
            for bt in bet_types:
                if text.startswith(bt) and bt not in payouts:
                    m = re.search(r"([\d,]+)円", text)
                    if m:
                        try:
                            payouts[bt] = int(m.group(1).replace(",", ""))
                        except ValueError:
                            pass
                    break
    return payouts


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
# 11. JRA Official Track Conditions (Ground Truth)
# ═══════════════════════════════════════════════════════════

# JRA公式: going → 標準クッション値・含水率マッピング
# 出典: JRA馬場情報基準値 (https://www.jra.go.jp/keiba/baba/)
_JRA_GOING_STANDARDS = {
    "良":   {"cushion": 10.0, "moisture_goal": 11.0, "moisture_4c": 11.5},
    "稍重": {"cushion":  8.0, "moisture_goal": 14.5, "moisture_4c": 15.0},
    "重":   {"cushion":  6.0, "moisture_goal": 18.5, "moisture_4c": 19.0},
    "不良": {"cushion":  4.0, "moisture_goal": 23.0, "moisture_4c": 23.5},
}

# JRA会場コード (URL用)
JRA_VENUE_CODES = {
    "東京": "tokyo", "中山": "nakayama", "阪神": "hanshin",
    "京都": "kyoto", "中京": "chukyo", "小倉": "kokura",
    "新潟": "niigata", "函館": "hakodate", "札幌": "sapporo", "福島": "fukushima",
}

# JRAサイト用ヘッダー (ブラウザ偽装)
_JRA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
    "Referer": "https://www.jra.go.jp/",
}


def fetch_jra_track_conditions(venue: str, race_date: date) -> dict:
    """
    JRA公式サイトから馬場情報をGround Truthとして取得。
    取得失敗時は going から JRA公式基準値を逆算してフォールバック。

    Returns:
      cushion_value:       str   クッション値 (例: "9.2")
      water_content_goal:  str   ゴール前含水率 (例: "11.5%")
      water_content_4c:    str   4コーナー含水率 (例: "12.1%")
      track_bias_text:     str   馬場傾向テキスト
      going:               str   馬場状態 (良/稍重/重/不良)
      inner_rail_moved:    bool  内柵移動の有無
      turf_replaced:       bool  芝張り替えの有無
      is_fallback:         bool  フォールバック値かどうか
      source:              str
    """
    result = {
        "cushion_value": "",
        "water_content_goal": "",
        "water_content_4c": "",
        "track_bias_text": "",
        "going": "",
        "inner_rail_moved": False,
        "turf_replaced": False,
        "is_fallback": False,
        "source": "JRA公式",
        "fetched_at": datetime.now().isoformat(),
    }

    # ① JRA baba/condition/ — 開催当日の馬場状態テーブル
    _try_jra_url("https://www.jra.go.jp/keiba/baba/condition/", venue, result)

    # ② netkeiba shutuba AJAX — JRA公式を反映した最も信頼性高いHTML
    if not result["cushion_value"]:
        date_str = race_date.strftime("%Y%m%d")
        soup = _get(
            "https://race.netkeiba.com/top/race_list_sub.html",
            params={"kaisai_date": date_str},
            encoding="utf-8",
        )
        if soup:
            _parse_netkeiba_track(soup, venue, result)

    # ③ JRA thisweek — 静的HTML (EUC-JP)
    if not result["cushion_value"]:
        _try_jra_url("https://www.jra.go.jp/keiba/thisweek/", venue, result,
                     encoding="euc-jp")

    # ④ 取得失敗 → going から JRA公式基準値でフォールバック
    if not result["cushion_value"] and result["going"]:
        _apply_going_fallback(result)
    elif not result["cushion_value"] and not result["going"]:
        # going も不明なら "良" 基準で警告フォールバック
        result["going"] = "良(推定)"
        _apply_going_fallback(result, going_key="良")
        print(f"[JRA scraper] WARNING: going不明。良(推定)の標準値を使用。venue={venue}")

    return result


def _try_jra_url(url: str, venue: str, result: dict, encoding: str = "utf-8"):
    """指定JRA URLから馬場情報を抽出。"""
    try:
        import requests as _req
        r = _req.get(url, headers=_JRA_HEADERS, timeout=10)
        r.raise_for_status()
        r.encoding = encoding
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        # 会場セクションに絞り込み
        venue_text = text
        for name in JRA_VENUE_CODES:
            if name in venue and name in text:
                # 会場名以降の300文字を抽出
                idx = text.find(name)
                venue_text = text[idx:idx + 600]
                break

        _extract_jra_track_values(venue_text, result)
        _extract_bias_info(text, result)
    except Exception as e:
        print(f"[JRA scraper] WARNING: {url} 取得失敗 ({e})")


def _parse_netkeiba_track(soup: "BeautifulSoup", venue: str, result: dict):
    """netkeiba race_list_sub から馬場情報を抽出 (JRA公式を反映)。"""
    for box in soup.select(".RaceList_Box"):
        venue_tag = box.select_one(".RaceList_DataTitle")
        if not venue_tag:
            continue
        if venue and venue not in venue_tag.get_text():
            continue
        text = box.get_text(" ", strip=True)
        _extract_jra_track_values(text, result)
        _extract_bias_info(text, result)
        break


def _extract_jra_track_values(text: str, result: dict):
    """正規表現でクッション値・含水率・馬場状態を抽出。"""
    # クッション値: 「クッション値9.2」「クッション値：9.2」「CV9.2」
    if not result["cushion_value"]:
        for pat in [
            r"クッション値[：:\s]*(\d+(?:\.\d+)?)",
            r"クッション[：:\s]*(\d+(?:\.\d+)?)",
            r"\bCV[：:\s]*(\d+(?:\.\d+)?)",
            r"cushion[：:\s]*(\d+(?:\.\d+)?)",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                result["cushion_value"] = m.group(1)
                break

    # 含水率 ゴール前: 「ゴール前12.3%」「含水率 ゴール前 12.3」
    if not result["water_content_goal"]:
        for pat in [
            r"ゴール前[^%\d]*(\d+(?:\.\d+)?)\s*%",
            r"ゴール前[：:\s]*(\d+(?:\.\d+)?)",
            r"含水率[^%\d\n]{0,6}(\d+(?:\.\d+)?)\s*%",
        ]:
            m = re.search(pat, text)
            if m:
                result["water_content_goal"] = m.group(1) + "%"
                break

    # 含水率 4コーナー
    if not result["water_content_4c"]:
        for pat in [
            r"4[コか]ーナー[^%\d]*(\d+(?:\.\d+)?)\s*%",
            r"4[コか]ーナー[：:\s]*(\d+(?:\.\d+)?)",
            r"4角[^%\d]*(\d+(?:\.\d+)?)\s*%",
        ]:
            m = re.search(pat, text)
            if m:
                result["water_content_4c"] = m.group(1) + "%"
                break

    # 馬場状態
    if not result["going"]:
        for going in ["不良", "重", "稍重", "良"]:
            if going in text:
                result["going"] = going
                break


def _extract_bias_info(text: str, result: dict):
    """馬場傾向・内柵移動・芝張替フラグを抽出。"""
    bias_kws = ["内ラチ", "内柵", "外ラチ", "外柵", "内有利", "外有利",
                "Aコース", "Bコース", "Cコース", "Dコース",
                "馬場傾向", "前有利", "差し有利", "追込有利"]
    sentences = re.split(r'[。．\n]', text)
    hits = [s.strip() for s in sentences if any(kw in s for kw in bias_kws) and len(s) > 5]
    if hits and not result["track_bias_text"]:
        result["track_bias_text"] = "。".join(hits[:3])

    if any(kw in text for kw in ["内ラチ移動", "内柵移動", "ラチ移動"]):
        result["inner_rail_moved"] = True
    if any(kw in text for kw in ["芝張替", "張り替え", "張替え", "新芝"]):
        result["turf_replaced"] = True


def _apply_going_fallback(result: dict, going_key: str = ""):
    """
    going → JRA公式基準値でクッション値・含水率を設定。
    is_fallback=True をセットして区別可能にする。
    """
    key = going_key or result.get("going", "良")
    # 稍重 など末尾が変わるケースを正規化
    for k in ["不良", "重", "稍重", "良"]:
        if k in key:
            key = k
            break
    std = _JRA_GOING_STANDARDS.get(key, _JRA_GOING_STANDARDS["良"])
    result["cushion_value"] = str(std["cushion"])
    result["water_content_goal"] = f"{std['moisture_goal']}%"
    result["water_content_4c"] = f"{std['moisture_4c']}%"
    result["is_fallback"] = True
    result["source"] = f"JRA公式基準値(フォールバック: {key})"
    print(f"[JRA scraper] FALLBACK: going={key} → CV={std['cushion']}, "
          f"moisture={std['moisture_goal']}%")


def fetch_jra_race_changes(race_id: str) -> dict:
    """
    netkeiba shutuba から JRA公式反映の出走取消・天候/馬場状態変更を取得。

    Returns:
      scratched:    list[str]  取消馬名リスト
      going_change: str        最新馬場状態
      weather:      str        最新天候
    """
    result = {"scratched": [], "going_change": "", "weather": "", "source": "netkeiba(JRA反映)"}

    soup = _get(
        f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}",
        encoding="utf-8",
    )
    if not soup:
        return result

    text = soup.get_text(" ", strip=True)

    # 馬場状態
    for going in ["不良", "重", "稍重", "良"]:
        m = re.search(rf"馬場[：:\s]*({going})", text)
        if m:
            result["going_change"] = m.group(1)
            break

    # 天候
    for pat in [r"天候[：:\s]*([晴曇雨雪小]+)", r"天気[：:\s]*([晴曇雨雪小]+)"]:
        m = re.search(pat, text)
        if m:
            result["weather"] = m.group(1)
            break

    # 出走取消馬
    for row in soup.select("tr.HorseList"):
        row_text = row.get_text(" ", strip=True)
        if "取消" in row_text or "除外" in row_text:
            name_tag = row.select_one(".HorseName a")
            if name_tag:
                result["scratched"].append(name_tag.get_text(strip=True))

    return result


# ─────────────────────────────────────────────
# KeibaLab — Live Paddock & Weight Data
# ─────────────────────────────────────────────

def fetch_keibalab_horse_weights(race_id: str, horse_names: list) -> dict:
    """
    KeibaLabのレースページから当日馬体重・増減・パドック短評を取得。

    race_id: netkeiba形式 (例: 202506010101)
    Returns:
      {horse_name: {"weight": int, "change": int, "comment": str}}
    """
    # KeibaLab race_id: YYYYMMDDXXXX → date(8桁) + 4桁会場+レース番号
    date_part = race_id[:8]
    venue_race = race_id[8:] if len(race_id) > 8 else ""

    # KeibaLab URL: /db/race/YYYYMMDDXXXX/
    kl_url = f"https://www.keibalab.jp/db/race/{date_part}{venue_race}/"
    soup = _get(kl_url, encoding="utf-8", delay=1.2)

    results = {}
    if not soup:
        return results

    # 馬体重テーブル: td 内 "438(-14)" 形式
    for row in soup.select("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        row_text = row.get_text(" ", strip=True)

        # 馬体重パターン: 数字3-4桁 + (±数字)
        wm = re.search(r"(\d{3,4})\(([+-]?\d+)\)", row_text)
        if not wm:
            continue

        weight_kg = int(wm.group(1))
        change_kg = int(wm.group(2))

        # 馬名を探す (行内のリンクテキスト or セル)
        name_tag = row.select_one("a")
        if not name_tag:
            continue
        cell_name = name_tag.get_text(strip=True)

        # horse_names との突合
        matched = next((n for n in horse_names if n in cell_name or cell_name in n), None)
        if matched:
            results[matched] = {
                "weight": weight_kg,
                "change": change_kg,
                "comment": "",  # paddock comment (後続で取得)
            }

    # パドック短評: .paddock, .comment 等のクラスを試行
    for item in soup.select(".paddock, .paddock_comment, .horse_comment, .comment"):
        item_text = item.get_text(strip=True)
        for name in horse_names:
            if name in item_text and name in results:
                results[name]["comment"] = item_text
                break

    return results


# ═══════════════════════════════════════════════════════════
# 12. Paddock reports — multi-source (race day SNS/news)
# ═══════════════════════════════════════════════════════════

def fetch_paddock_reports(race_id: str, horse_names: list, race_name: str = "") -> dict:
    """
    当日のパドック情報を複数の公開ソースから取得。
    Sources (priority order):
      1. netkeiba パドックページ
      2. Yahoo!ニュース検索 (当日記事)
      3. uma-jo.jp
      4. keibago.com

    Returns:
      {horse_name: {"text": str, "source": str, "scores": dict}}
    """
    reports: dict = {}

    # 1. netkeiba paddock
    _scrape_netkeiba_paddock(race_id, horse_names, reports)

    # 2. Yahoo!ニュース (SNS的な当日速報)
    missing = [n for n in horse_names if not reports.get(n, {}).get("text")]
    if missing:
        _scrape_yahoo_news_paddock(race_name or race_id, missing, reports)

    # 3. uma-jo.jp
    missing = [n for n in horse_names if not reports.get(n, {}).get("text")]
    if missing:
        _scrape_umajo_paddock(race_id, missing, reports)

    # 4. keibago.com
    missing = [n for n in horse_names if not reports.get(n, {}).get("text")]
    if missing:
        _scrape_keibago_paddock(race_id, missing, reports)

    # 未取得馬は空エントリ
    for name in horse_names:
        if name not in reports:
            reports[name] = {"text": "", "source": "", "scores": {}}

    return reports


def _scrape_netkeiba_paddock(race_id: str, horse_names: list, reports: dict):
    """netkeiba paddock.html からパドックコメントを取得。"""
    soup = _get(
        "https://race.netkeiba.com/race/paddock.html",
        params={"race_id": race_id},
        encoding="utf-8",
    )
    if not soup:
        return

    full_text = soup.get_text(" ", strip=True)

    # horse-by-horse structured comments
    for sel in [".PaddockComment", ".Paddock_Comment", ".HorsePaddock",
                ".PaddockData_Item", ".paddock_comment"]:
        for item in soup.select(sel):
            name_tag = item.select_one(".HorseName, .Horse_Name, .name")
            comment_tag = item.select_one(".Comment, .PaddockText, p, .text")
            if not (name_tag and comment_tag):
                continue
            item_name = _clean_text(name_tag.get_text(strip=True))
            text = _clean_text(comment_tag.get_text(strip=True))
            for hn in horse_names:
                if hn in item_name or item_name in hn:
                    reports[hn] = {
                        "text": text,
                        "source": "netkeiba",
                        "scores": parse_paddock_comment(text),
                    }

    # フォールバック: ページ全体テキストから馬名周辺の文を抽出
    if not any(reports.get(n, {}).get("text") for n in horse_names):
        for name in horse_names:
            sentences = [_clean_text(s) for s in re.split(r'[。．\n]', full_text)
                         if name in s and len(s) > 10]
            if sentences:
                combined = "。".join(sentences[:3])
                reports[name] = {
                    "text": combined,
                    "source": "netkeiba(全文抽出)",
                    "scores": parse_paddock_comment(combined),
                }


def _scrape_yahoo_news_paddock(race_name: str, horse_names: list, reports: dict):
    """Yahoo!ニュース競馬検索で当日パドック記事を取得。"""
    query = f"{race_name} パドック"
    soup = _get(
        "https://news.yahoo.co.jp/search",
        params={"p": query, "ei": "utf-8"},
        delay=1.5,
    )
    if not soup:
        return

    # 全記事テキストを収集
    snippets = []
    for el in soup.select(".newsFeed_item_detail, .sc-fzoLsD, .article_body, p"):
        t = el.get_text(strip=True)
        if len(t) > 20:
            snippets.append(t)

    full_text = "。".join(snippets)

    for name in horse_names:
        sentences = [_clean_text(s) for s in re.split(r'[。．\n]', full_text)
                     if name in s and len(s) > 10]
        if sentences:
            combined = "。".join(sentences[:4])
            reports[name] = {
                "text": combined,
                "source": "Yahoo!ニュース",
                "scores": parse_paddock_comment(combined),
            }


def _scrape_umajo_paddock(race_id: str, horse_names: list, reports: dict):
    """uma-jo.jp から当日パドックレポートを取得。"""
    # uma-jo はレース日付ベースのURL構造
    date_part = race_id[:8] if len(race_id) >= 8 else ""
    if not date_part:
        return

    year, month, day = date_part[:4], date_part[4:6], date_part[6:8]
    soup = _get(
        f"https://uma-jo.jp/{year}/{month}/{day}/",
        delay=1.2,
    )
    if not soup:
        return

    full_text = soup.get_text(" ", strip=True)
    for name in horse_names:
        sentences = [_clean_text(s) for s in re.split(r'[。．\n]', full_text)
                     if name in s and len(s) > 10]
        if sentences:
            combined = "。".join(sentences[:3])
            reports[name] = {
                "text": combined,
                "source": "uma-jo.jp",
                "scores": parse_paddock_comment(combined),
            }


def _scrape_keibago_paddock(race_id: str, horse_names: list, reports: dict):
    """keibago.com から当日パドック情報を取得。"""
    soup = _get(
        f"https://keibago.com/race/{race_id}/paddock/",
        delay=1.2,
    )
    if not soup:
        return

    full_text = soup.get_text(" ", strip=True)
    for name in horse_names:
        sentences = [_clean_text(s) for s in re.split(r'[。．\n]', full_text)
                     if name in s and len(s) > 10]
        if sentences:
            combined = "。".join(sentences[:3])
            reports[name] = {
                "text": combined,
                "source": "keibago.com",
                "scores": parse_paddock_comment(combined),
            }


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


def fetch_past_g_races(n_weeks: int = 4) -> list[dict]:
    """
    直近N週分の終了済みG1/G2/G3レース一覧を取得する。
    土曜・日曜の両日を対象に、今日より前の開催分のみ返す。

    Returns:
        list of race dicts (race_id, race_name, grade, venue, race_date, ...)
    """
    from datetime import timedelta as _td
    today = date.today()
    weekday = today.weekday()  # 月=0 … 日=6

    # 直近の土曜（今日が土曜なら今日を含む）
    days_since_sat = (weekday - 5) % 7
    last_sat = today - _td(days=days_since_sat)

    races: list[dict] = []
    seen: set = set()

    for w in range(n_weeks):
        for day_delta in (0, 1):  # 土=0, 日=1
            target = last_sat - _td(weeks=w) + _td(days=day_delta)
            # 今日以降（未来）はスキップ
            if target >= today:
                continue
            try:
                day_races = fetch_race_list_netkeiba(target)
            except Exception:
                day_races = []
            for r in day_races:
                if r["race_id"] not in seen:
                    seen.add(r["race_id"])
                    races.append(r)
        time.sleep(REQUEST_DELAY)

    # 日付降順（新しい順）で返す
    races.sort(key=lambda x: x.get("race_date", ""), reverse=True)
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
