PAIRS = [('A-B', 'AとB'), ('B-C', 'BとC'), ('C-A', 'CとA')]
THEMES = ['無人島ー縛りあり', '無人島ー縛りなし', 'AIー縛りあり', 'AIー縛りなし',
          '外国人受け入れー縛りあり', '外国人受け入れー縛りなし']
HEADERS = ['タイムスタンプ', 'ID', '割り当て', 'テーマ'] + [f'{i}. {label}' for i in range(1,55) for _,label in PAIRS] + ['感想']
