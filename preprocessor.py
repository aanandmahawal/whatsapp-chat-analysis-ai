import re
import pandas as pd


def _parse_dates(date_strings, date_format):
    """
    WhatsApp writes dates as DD/MM/YY or MM/DD/YY depending on the phone's
    locale (India -> day-first, US English -> month-first). The export gives no
    hint which one it is, so try both and keep the order that parses cleanly.
    Tie-break: a chat export is chronological, so prefer the order whose
    timestamps never go backwards.
    """
    # Normalise the narrow no-break space some phones put before AM/PM.
    s = date_strings.astype(str).str.replace('\u202f', ' ', regex=False).str.replace('\u00a0', ' ', regex=False)
    day_first = pd.to_datetime(s, format=date_format, errors='coerce')
    month_first = pd.to_datetime(s, format=date_format.replace('%d/%m', '%m/%d'), errors='coerce')

    def score(parsed):
        valid = parsed.dropna()
        backwards = int((valid.diff() < pd.Timedelta(0)).sum()) if len(valid) > 1 else 0
        return (int(parsed.isna().sum()), backwards)   # lower is better on both

    return month_first if score(month_first) < score(day_first) else day_first


def preprocess(data):
    # Try matching both formats
    pattern_12hr = r'\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}[\u2000-\u206F\s]*[AaPp][Mm] - '
    pattern_24hr = r'\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} - '

    # Choose pattern based on presence in data
    if re.search(pattern_12hr, data):
        pattern = pattern_12hr
        date_format = "%d/%m/%y, %I:%M %p"
    else:
        pattern = pattern_24hr
        date_format = "%d/%m/%y, %H:%M"

    # Extract date strings and messages
    message_list = re.split(pattern, data)[1:]
    date_list = re.findall(pattern, data)

    if len(message_list) != len(date_list):
        raise ValueError(f"Mismatch between dates and messages: {len(date_list)} vs {len(message_list)}")

    # Remove trailing " - " from dates
    date_list = [d.strip().replace(" -", "") for d in date_list]

    # Create DataFrame
    df = pd.DataFrame({'user_message': message_list, 'message_date': date_list})
    df['message_date'] = _parse_dates(df['message_date'], date_format)
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Separate users and messages
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append("group_notification")
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract features
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Time periods
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append("23-00")
        elif hour == 0:
            period.append("00-01")
        else:
            period.append(f"{str(hour).zfill(2)}-{str(hour + 1).zfill(2)}")
    df['period'] = period

    return df
