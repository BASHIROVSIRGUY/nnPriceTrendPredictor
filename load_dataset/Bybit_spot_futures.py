import os
import csv
import time
import datetime as dt
import requests
import glob
import sys


BASE_URL = "https://api.bybit.com"
KLINES_ENDPOINT = "/v5/market/kline"
INSTRUMENTS_ENDPOINT = "/v5/market/instruments-info"
BYBIT_CATEGORY = os.getenv("BYBIT_CATEGORY", "linear")  # "linear" — USDT perpetual фьючерсы, "spot" — спот


def minutes_to_interval(minutes):
    mapping = {
        1: "1", 3: "3", 5: "5", 15: "15", 30: "30",
        60: "60", 120: "120", 240: "240", 360: "360", 480: "480",
        720: "720", 1440: "D"}
    if minutes not in mapping:
        raise ValueError(f"Неподдерживаемый таймфрейм: {minutes} минут")
    return mapping[minutes]
def interval_ms(interval):
    mapping = {
        "1": 60_000, "3": 180_000, "5": 300_000, "15": 900_000, "30": 1_800_000,
        "60": 3_600_000, "120": 7_200_000, "240": 14_400_000, "360": 21_600_000,
        "480": 28_800_000, "720": 43_200_000, "D": 86_400_000}
    return mapping[interval]
def fmt_ts(ts_ms):
    try:
        return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return str(ts_ms)
def _to_ms(value):
    try:
        v = int(value)
        return v if v >= 10**12 else v * 1000
    except Exception:
        return None
def get_listing_ms(symbol, category):
    try:
        r = requests.get(BASE_URL + INSTRUMENTS_ENDPOINT, params={"category": category, "symbol": symbol}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict) or data.get("retCode") != 0:
            return None
        items = (data.get("result") or {}).get("list") or []
        if not items:
            return None
        info = items[0]
        candidates = [
            _to_ms(info.get("listTime")),
            _to_ms(info.get("listingTime")),
            _to_ms(info.get("launchTime")),
            _to_ms(info.get("createdTime")),
        ]
        candidates = [x for x in candidates if x]
        return min(candidates) if candidates else None
    except Exception:
        return None
def read_existing_csv(file_path):
    if not os.path.exists(file_path):
        return []
    rows = []
    with open(file_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                int(row[0])
            except Exception:
                continue
            rows.append(row)
    return rows
def write_csv(file_path, rows):
    headers = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
def find_periods_to_fetch(existing, want_start, want_end, step_ms):
    if want_start > want_end:
        return []
    periods = []
    timestamps = sorted([int(r[0]) for r in existing]) if existing else []
    if not timestamps:
        return [(want_start, want_end)]
    first_ts = timestamps[0]
    last_ts = timestamps[-1]
    if want_start < first_ts:
        periods.append((want_start, first_ts - step_ms))
    for i in range(len(timestamps) - 1):
        cur_ts = timestamps[i]
        next_ts = timestamps[i + 1]
        expected = cur_ts + step_ms
        if next_ts > expected:
            gap_start = expected
            gap_end = next_ts - step_ms
            if gap_start <= want_end and gap_end >= want_start:
                periods.append((max(gap_start, want_start), min(gap_end, want_end)))
    tail_start = last_ts + step_ms
    if tail_start <= want_end:
        periods.append((max(tail_start, want_start), want_end))
    if not periods:
        return []
    periods = [p for p in periods if p[0] <= p[1]]
    periods.sort(key=lambda x: x[0])
    merged = []
    cs, ce = periods[0]
    for s, e in periods[1:]:
        if s <= ce + step_ms:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged
def fetch_klines(symbol, interval, start_ms, end_ms, category):
    result = []
    step = interval_ms(interval)
    listing_ms = get_listing_ms(symbol, category)
    cur = max(start_ms, listing_ms) if listing_ms else start_ms
    if listing_ms and start_ms < listing_ms:
        print(f"    ℹ Начинаю с даты листинга инструмента: {fmt_ts(cur)}")
    while cur <= end_ms:
        chunk_end = min(end_ms, cur + step * 1000 - step)
        print(f"  → Запрос {symbol} {interval} ({category}): {fmt_ts(cur)} → {fmt_ts(chunk_end)}")
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "start": cur,
            "end": chunk_end,
            "limit": 1000}
        try:
            r = requests.get(BASE_URL + KLINES_ENDPOINT, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception:
            print("    ! Ошибка сети/API, окно пропущено")
            cur = chunk_end + step
            continue
        if not isinstance(data, dict) or data.get("retCode") != 0:
            print(f"    ! Ошибка ответа API, окно пропущено: {data.get('retMsg') if isinstance(data, dict) else 'unknown'}")
            cur = chunk_end + step
            continue
        lst = (data.get("result") or {}).get("list") or []
        if not lst:
            print("    ! Пустой ответ, окно пропущено")
            cur = chunk_end + step
            continue
        try:
            lst.sort(key=lambda x: int(x[0]))
        except Exception:
            pass

        rows_batch = []
        for it in lst:
            # Формат элемента: [open_time, open, high, low, close, volume, turnover]
            try:
                open_ts = int(it[0])
                open_p, high_p, low_p, close_p = str(it[1]), str(it[2]), str(it[3]), str(it[4])
                volume_base = str(it[5])
                turnover_quote = str(it[6])
            except Exception:
                continue
            row7 = [str(open_ts), open_p, high_p, low_p, close_p, volume_base, turnover_quote,]
            rows_batch.append(row7)
        if not rows_batch:
            print("    ! Нечего добавлять из этого окна, пропускаю")
            cur = chunk_end + step
            continue

        print(f"    ✓ Получено чанком: {len(rows_batch)} свечей, последняя: {fmt_ts(int(rows_batch[-1][0]))}")
        result.extend(rows_batch)
        last_open = int(rows_batch[-1][0])
        if last_open < cur:
            cur = chunk_end + step
            continue
        cur = last_open + step
        time.sleep(0.1)
    return result
def merge_and_dedup(existing, new_rows):
    all_rows = []
    # Нормализуем существующие строки к 7 столбцам
    for r in existing:
        if not r:
            continue
        try:
            open_ts = str(r[0])
            open_p = str(r[1]) if len(r) > 1 else "0"
            high_p = str(r[2]) if len(r) > 2 else "0"
            low_p = str(r[3]) if len(r) > 3 else "0"
            close_p = str(r[4]) if len(r) > 4 else "0"
            volume_base = str(r[5]) if len(r) > 5 else "0"
            # Если старая запись имела 12 колонок, turnover был в колонке 7 (индекс 7)
            turnover_quote = str(r[6]) if len(r) == 7 else (str(r[7]) if len(r) > 7 else "0")
        except Exception:
            continue
        all_rows.append([open_ts, open_p, high_p, low_p, close_p, volume_base, turnover_quote])

    # Новые строки уже 7-колоночные
    for r in new_rows:
        if not r:
            continue
        # Жестко приводим к 7 строковым полям
        rr = [str(x) for x in r[:7]]
        while len(rr) < 7:
            rr.append("0")
        all_rows.append(rr)

    all_rows.sort(key=lambda x: int(x[0]))
    unique = []
    seen = set()
    for row in all_rows:
        ts = row[0]
        if ts in seen:
            continue
        seen.add(ts)
        unique.append(row)
    return unique
def find_best_base_timeframe(target_minutes):
    """Находит оптимальный базовый ТФ для агрегации в целевой ТФ"""
    supported = [1, 5, 15, 30, 60, 120, 240, 360, 480, 720, 1440]
    # Ищем наибольший делитель из поддерживаемых
    best_base = None
    for base in reversed(supported):
        if target_minutes % base == 0:
            best_base = base
            break
    if best_base is None:
        raise ValueError(f"Невозможно составить ТФ {target_minutes} из поддерживаемых интервалов")
    multiplier = target_minutes // best_base
    return best_base, multiplier
def aggregate_candles(rows, multiplier, target_step_ms):
    """Агрегирует свечи: multiplier свечей -> 1 свеча"""
    if multiplier == 1:
        return rows
    rows_sorted = sorted(rows, key=lambda x: int(x[0]))
    aggregated = []
    i = 0
    while i < len(rows_sorted):
        # Берем первую свечу группы
        first_row = rows_sorted[i]
        open_ts = int(first_row[0])
        open_price = float(first_row[1])
        high_price = float(first_row[2])
        low_price = float(first_row[3])
        volume_sum = float(first_row[5])
        turnover_sum = float(first_row[6])
        close_price = float(first_row[4])
        j = 1
        while j < multiplier and i + j < len(rows_sorted):
            next_row = rows_sorted[i + j]
            next_ts = int(next_row[0])
            # Проверяем, что свечи идут подряд
            expected_ts = open_ts + j * (target_step_ms // multiplier)
            if abs(next_ts - expected_ts) > 60000:  # допуск 1 минута
                break
            high_price = max(high_price, float(next_row[2]))
            low_price = min(low_price, float(next_row[3]))
            close_price = float(next_row[4])
            volume_sum += float(next_row[5])
            turnover_sum += float(next_row[6])
            j += 1
        
        # Добавляем агрегированную свечу только если собрали полную группу
        if j == multiplier:
            aggregated.append([
                str(open_ts),
                str(open_price),
                str(high_price),
                str(low_price),
                str(close_price),
                str(volume_sum),
                str(turnover_sum)])
        i += j
    return aggregated
def process_custom_tf(coins, timeframe_minutes, start_date, end_date, base_folder=None):
    """Обработка произвольных ТФ через агрегацию базовых ТФ"""
    if base_folder is None:
        base_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
    # Определяем базовый ТФ и множитель
    base_tf, multiplier = find_best_base_timeframe(timeframe_minutes)
    print(f"Кастомный ТФ {timeframe_minutes}m = {multiplier} × {base_tf}m")
    # Папки для базового и целевого ТФ
    base_dir = os.path.join(base_folder, f"bybit_tkns_{base_tf}")
    target_dir = os.path.join(base_folder, f"bybit_tkns_{timeframe_minutes}")
    os.makedirs(target_dir, exist_ok=True)
    target_step_ms = timeframe_minutes * 60_000
    # Сначала загружаем базовые данные через обычный process
    print(f"\n[1/2] Загрузка базовых данных ({base_tf}m)...")
    process(coins, base_tf, start_date, end_date, base_folder)
    # Теперь агрегируем
    print(f"\n[2/2] Агрегация в {timeframe_minutes}m...")
    num = 0
    for coin in coins:
        num += 1
        coin_u = coin.upper()
        base_file = os.path.join(base_dir, f"{coin_u}_{base_tf}.csv")
        target_file = os.path.join(target_dir, f"{coin_u}_{timeframe_minutes}.csv")
        print(f"\n{num}/{len(coins)}  {coin_u}")
        if not os.path.exists(base_file):
            print(f"  ! Базовый файл не найден: {base_file}")
            continue
        # Читаем базовые свечи
        base_rows = read_existing_csv(base_file)
        if not base_rows:
            print(f"  ! Нет данных в базовом файле")
            continue
        print(f"  → Базовых свечей: {len(base_rows)}")
        # Агрегируем
        aggregated = aggregate_candles(base_rows, multiplier, target_step_ms)
        print(f"  → Агрегированных свечей: {len(aggregated)}")
        if aggregated:
            write_csv(target_file, aggregated)
            print(f"  ✓ Сохранено в {target_file}")
def process(coins, timeframe_minutes, start_date, end_date, base_folder=None):
    if base_folder is None:
        base_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
    interval = minutes_to_interval(timeframe_minutes)
    step = interval_ms(interval)
    tokens_dir = os.path.join(base_folder, f"bybit_tkns_{timeframe_minutes}")
    os.makedirs(tokens_dir, exist_ok=True)
    print(f"Папка вывода: {os.path.abspath(tokens_dir)}")
    want_start = int(dt.datetime.strptime(start_date, "%d.%m.%Y").replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    d = dt.datetime.strptime(end_date, "%d.%m.%Y")
    want_end = int(dt.datetime(d.year, d.month, d.day, 23, 59, 59, 999000, tzinfo=dt.timezone.utc).timestamp() * 1000)
    num = 0
    for coin in coins:
        num += 1
        coin_u = coin.upper()
        exact_path = os.path.join(tokens_dir, f"{coin_u}_{timeframe_minutes}.csv")
        candidates = sorted(glob.glob(os.path.join(tokens_dir, f"{coin_u}*.csv")))
        print(f"\n{num}/{len(coins)}  {coin_u}")
        if os.path.exists(exact_path):
            file_path = exact_path
        elif candidates:
            file_path = candidates[0]
        else:
            file_path = exact_path

        existing = []
        for p in candidates:
            existing.extend(read_existing_csv(p))
        periods = find_periods_to_fetch(existing, want_start, want_end, step)
        if periods:
            print("Периоды для дозагрузки:")
            for i, (s, e) in enumerate(periods, 1):
                print(f"  [{i}] {fmt_ts(s)} → {fmt_ts(e)}")
        if not existing and not periods:
            continue
        for s, e in periods:
            kl = fetch_klines(coin_u, interval, s, e, BYBIT_CATEGORY)
            if not kl:
                continue
            existing = merge_and_dedup(existing, kl)
        if not os.path.exists(file_path) and not existing:
            continue
        try:
            existing.sort(key=lambda x: int(x[0]))
        except Exception:
            print(f"Ошибка сортировки: {file_path}")
            exit()
        write_csv(file_path, existing)


tokens = ['ADAUSDT', 'APTUSDT', 'AVAXUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT', 'ETHUSDT',
          'GALAUSDT', 'HBARUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT', 'UNIUSDT', 'XLMUSDT', 'XRPUSDT', 'ZECUSDT']
timeframes = [60]

# tokens = ['AAVEUSDT']
# timeframes = [60, 45, 30, 15, 5, 1]

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    end_date_str = dt.datetime.now().strftime("%d.%m.%Y")

    for i in range(1):
        for timeframe_minutes in timeframes:
            process(coins=tokens, timeframe_minutes=timeframe_minutes, start_date="01.01.2022", end_date=end_date_str)

    # Проверка пропусков
    test_gaps = 0
    if test_gaps:
        def check_gaps(file_path, tf_minutes):
            step = interval_ms(minutes_to_interval(tf_minutes))
            with open(file_path, "r", newline="") as f:
                rdr = csv.reader(f)
                times = []
                for row in rdr:
                    if not row:
                        continue
                    try:
                        t = int(row[0])
                    except Exception:
                        continue
                    times.append(t)
            times = sorted(set(times))
            gaps = []
            for i in range(len(times) - 1):
                cur, nxt = times[i], times[i + 1]
                if nxt - cur > step:
                    gaps.append((cur + step, nxt - step))
            if not gaps:
                print("Пропусков нет")
            else:
                print("Найдены пропуски:")
                for i, (s, e) in enumerate(gaps, 1):
                    print(f"  [{i}] {fmt_ts(s)} → {fmt_ts(e)}")
        for tf in [1, 5, 15, 30, 60, 240]:
            print(tf)
            for ticker in tokens:
                GAP_FILE = f"bybit_tkns_{tf}m/{ticker}_{tf}.csv"
                check_gaps(GAP_FILE, tf)
