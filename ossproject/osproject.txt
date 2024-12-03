import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


# CSV파일 읽어오기
df = pd.read_csv('male_players3.csv', dtype={108: str})

# 분석에 필요한 컬럼만 선택
columns_to_use = ['short_name', 'age', 'league_name', 'player_positions', 'potential', 'overall', 'height_cm', 'value_eur']
df = df[columns_to_use]

# 프리미어 리그 선수 데이터만 필터링
premier_league_df = df[df['league_name'] == 'Premier League'].copy()

# 선수의 주 포지션을 첫 번째 값으로 설정
premier_league_df['player_positions'] = premier_league_df['player_positions'].str.split(',').str[0]

# 대표 포지션으로 축약
position_mapping = {
    'GK': 'GK',
    'RB': 'DF', 'RWB': 'DF', 'CB': 'DF', 'LB': 'DF', 'LWB': 'DF',
    'CDM': 'MF', 'CM': 'MF', 'CAM': 'MF', 'RM': 'MF', 'LM': 'MF',
    'RW': 'FW', 'RF': 'FW', 'CF': 'FW', 'LF': 'FW', 'LW': 'FW', 'ST': 'FW'
}
premier_league_df['player_positions'] = premier_league_df['player_positions'].replace(position_mapping)

# 더 이상 필요하지 않은 컬럼 제거
premier_league_df = premier_league_df.drop('league_name', axis=1)

# groupby를 사용하여 데이터 그룹화 후 통계분석
agg_functions = ['mean', 'median', 'std', 'min', 'max']
position_stats = premier_league_df.groupby('player_positions')[
    ['potential', 'overall', 'age', 'height_cm', 'value_eur']
].agg(agg_functions)

# 통계 결과 출력
pd.options.display.float_format = '{:.2f}'.format
print(position_stats)

# 그래프 1. 전체 능력치와 시장 가치의 상관관계 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='overall', y='value_eur', data=premier_league_df,
    hue='player_positions', palette='viridis'
)
plt.title('Correlation between Overall and Market Value')
plt.xlabel('Overall')
plt.ylabel('Market Value')
plt.show()

# 그래프 2. 나이와 능력치의 관계 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(
    x='age', y='overall', data=premier_league_df,
    hue='player_positions', style='player_positions', palette='Set1'
)
plt.title('Overall by Age and Position')
plt.xlabel('Age')
plt.ylabel('Overall')
plt.show()

# 그래프 3. 전체 능력치 분포
plt.figure(figsize=(10, 6))
sns.histplot(premier_league_df['overall'], bins=20, kde=True)
plt.title('Distribution of Overall Rating')
plt.xlabel('Overall Rating')
plt.ylabel('Count')
plt.show()

# 그래프 4. 피어슨 상관계수 히트맵
plt.figure(figsize=(10, 8))
corr_matrix = premier_league_df[['potential', 'overall', 'age', 'height_cm', 'value_eur']].corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Pearson Correlation Matrix')
plt.show()

# 머신 러닝 모델 준비 (다항회귀분석)
# 결측값 제거
premier_league_df = premier_league_df.dropna()

# 입력 변수와 타겟 변수 설정
features = ['age', 'overall', 'potential', 'height_cm']
target = 'value_eur'
X = premier_league_df[features]
y = premier_league_df[target]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


degree = 2  # 다항식 차수 설정
poly = PolynomialFeatures(degree=degree, include_bias=False)

# 다항식 특징 변환
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 다항 회귀 모델 학습
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# 예측
y_pred_poly = poly_model.predict(X_test_poly)

# 성능 평가
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Polynomial Regression (Degree {degree}) - MAE: {mae_poly:.2f}, MSE: {mse_poly:.2f}, R²: {r2_poly:.2f}")


# 실제 값 vs 예측 값 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_poly, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel('Actual Market Value')
plt.ylabel('Predicted Market Value')
plt.title(f'Actual vs Predicted Market Value (Polynomial Regression, Degree {degree})')
plt.show()