import holidays

indo = holidays.Indonesia(years=2035)

for date, name in indo.items():
    print(date, name)