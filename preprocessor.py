import re
import pandas as pd

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
    df['message_date'] = pd.to_datetime(df['message_date'], format=date_format, errors='coerce')
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
