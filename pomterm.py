from prophet import Prophet
from prophet.diagnostics import cross_validation
import pandas as pd

class DataProcessor:
    def __init__(self):
        self.calendar = pd.read_csv('calendar.csv')
        self.sales_train_validation = pd.read_csv('sales_train_validation.csv')
        self.sales_train_evaluation = pd.read_csv('sales_train_evaluation.csv')
        self.sell_prices = pd.read_csv('sell_prices.csv')
        
        self.day_index = self.calendar["d"]
        self.event_type1 = self.calendar["event_type_1"]
        self.event_type2 = self.calendar["event_type_2"]
        self.snap_CA = self.calendar["snap_CA"]
        self.snap_TX = self.calendar["snap_TX"]
        self.snap_WI = self.calendar["snap_WI"]
        
        self.event_types = ['Cultural', 'National', 'Religious', 'Sporting']
        self.snap_types = ['snap_CA', 'snap_TX', 'snap_WI']
        
        self._process_events()
        
    def _process_events(self):
        # Process event types
        event_dfs = {}
        for event in self.event_types:
            mask = (self.calendar['event_type_1'] == event) | (self.calendar['event_type_2'] == event)
            days = self.calendar[mask]['date']
            df = pd.DataFrame({
                'holiday': event,
                'ds': days,
                'lower_window': 0,
                'upper_window': 1,
            })
            event_dfs[event] = df
            
        # Process snap types
        for snap in self.snap_types:
            mask = (self.calendar[snap] == 1)
            days = self.calendar[mask]['date']
            df = pd.DataFrame({
                'holiday': snap,
                'ds': days,
                'lower_window': 0,
                'upper_window': 1,
            })
            event_dfs[snap] = df  # Fixed: was using 'event' instead of 'snap'
        self.event_dfs = pd.concat(event_dfs.values())

    def get_sales_data(self, item_id, time_index=1799):
        """
        Get sales data for a specific item.
        Args:
            item_id: ID of the item to get sales data for
        Returns:
            DataFrame with dates and sales values
        """
        # Get row for this item
        item_data = self.sales_train_evaluation[self.sales_train_evaluation['id'] == item_id]
        
        if len(item_data) == 0:
            raise ValueError(f"Item ID {item_id} not found")
            
        # Get dates and values starting from column 6 (first sales column)
        dates = self.sales_train_evaluation.columns[6:]
        values = item_data.iloc[0, 6:].values
        
        # Create DataFrame with dates and sales values
        df = pd.DataFrame({
            'ds': dates,
            'y': values
        })

        return df

    def prepare_prophet_data(self):
        """
        Prepare data for Prophet model by converting column names to dates.
        """
        start_col = 6
        start_date = pd.to_datetime('2011-01-29')

        # Number of time-series columns (from d_1 to d_n)
        n_days = self.sales_train_evaluation.shape[1] - start_col

        # Generate new column names for the time-series part
        date_range = pd.date_range(start=start_date, periods=n_days)
        new_date_cols = date_range.strftime('%Y-%m-%d')

        # Create new column list
        self.sales_train_evaluation.columns = list(self.sales_train_evaluation.columns[:start_col]) + list(new_date_cols)

    def train_prophet_model(self, df):
        """
        Train Prophet model on given data.
        Args:
            df: DataFrame with 'ds' and 'y' columns
            events: Holiday events DataFrame
        Returns:
            Trained Prophet model
        """
        events = self.event_dfs
        model = Prophet(holidays=events)
        model.fit(df)
        return model

    def make_predictions(self, model, periods=560):
        """
        Make predictions using trained Prophet model.
        Args:
            model: Trained Prophet model
            periods: Number of days to forecast
        Returns:
            Forecast DataFrame
        """
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def predict_all_items(self, time_index=1799, num_items=None):
        """
        Make predictions for all items in the dataset.
        Args:
            time_index: Index for time series data
            num_items: Number of items to process (None for all items)
        Returns:
            DataFrame with predictions for all items
        """
        # 시작 날짜와 종료 날짜 설정
        start_date = pd.to_datetime('2011-01-29')  # 데이터 시작일
        train_end_date = pd.to_datetime('2016-06-19')  # 훈련 종료일
        forecast_end_date = pd.to_datetime('2017-12-31')  # 예측 종료일
        
        # 전체 날짜 범위 생성
        all_dates = pd.date_range(start=start_date, end=forecast_end_date)
        print(f"전체 날짜 수: {len(all_dates)}")
        
        # 날짜 열만 있는 최종 결과용 DataFrame 생성
        result_df = pd.DataFrame({'ds': all_dates})
        
        # 훈련 데이터 날짜 범위 (2011-01-29부터 2016-06-19까지)
        train_dates = all_dates[all_dates <= train_end_date]
        print(f"훈련 데이터 날짜 수: {len(train_dates)}")
        
        # 훈련 데이터 가져오기
        train_df = self.sales_train_evaluation.iloc[:, 6:]
        
        # 처리할 아이템 선택
        items_to_process = self.sales_train_evaluation.head(num_items) if num_items else self.sales_train_evaluation
        
        # 전체 아이템 수 파악
        total_items = len(items_to_process)
        print(f"전체 {total_items}개 아이템에 대해 예측을 시작합니다.")
        
        # 진행 상황 추적용 변수
        start_time = pd.Timestamp.now()
        success_count = 0
        error_count = 0
        
        # 각 아이템에 대해 예측 수행
        for index, rows in items_to_process.iterrows():
            try:
                # 진행률 계산 및 표시
                progress = (index + 1) / total_items * 100
                elapsed_time = (pd.Timestamp.now() - start_time).total_seconds() / 60  # 분 단위
                
                # 아이템 이름 가져오기
                name = rows['id']
                print(f"[{progress:.1f}% | {index+1}/{total_items} | {elapsed_time:.1f}분 경과] 아이템: {name} 예측 중...")
                
                # 날짜 열에서 훈련 기간에 해당하는 값 찾기
                train_date_cols = [col for col in train_df.columns if pd.to_datetime(col) <= train_end_date]
                
                if not train_date_cols:
                    print(f"  아이템 {name}: 훈련 데이터가 없습니다.")
                    error_count += 1
                    continue
                
                # 훈련 데이터 생성
                item_values = rows[train_date_cols].values
                train_data = pd.DataFrame({
                    'ds': pd.to_datetime(train_date_cols),
                    'y': item_values
                })
                
                # Prophet 모델 학습 (train_prophet_model 사용)
                m = self.train_prophet_model(train_data)
                
                # 예측 기간 설정 (2017년 12월 31일까지)
                future = m.make_future_dataframe(periods=len(result_df), freq='D')
                future = future[future['ds'] <= '2017-12-31']
                
                # 예측 수행
                forecast = m.predict(future)
                
                # 예측 결과 저장
                forecast_series = pd.Series(forecast['yhat'].values, index=pd.to_datetime(forecast['ds']))
                
                # 결과 DataFrame에 예측값 추가
                result_df[name] = result_df['ds'].map(lambda x: forecast_series.get(x) if x in forecast_series.index else None)
                
                # 메모리 최적화 - 불필요한 객체 명시적 삭제
                del m, forecast, forecast_series
                
                success_count += 1
                
                # 중간 저장 (3000개 아이템마다)
                if (index + 1) % 3000 == 0:
                    temp_result = result_df.copy()
                    temp_result['ds'] = temp_result['ds'].astype(str)
                    temp_result.to_csv(f"prediction_df_checkpoint_{index+1}.csv", index=False)
                    print(f"  중간 결과 저장 완료: prediction_df_checkpoint_{index+1}.csv")
                
            except Exception as e:
                print(f"  아이템 {name} 예측 중 오류 발생: {str(e)}")
                error_count += 1
                import traceback
                traceback.print_exc()
        
        # 예측 완료 후 통계 출력
        end_time = pd.Timestamp.now()
        total_time = (end_time - start_time).total_seconds() / 60  # 분 단위
        
        print("\n" + "="*50)
        print(f"예측 완료 통계:")
        print(f"- 전체 아이템: {total_items}개")
        print(f"- 성공: {success_count}개")
        print(f"- 실패: {error_count}개")
        print(f"- 소요 시간: {total_time:.1f}분")
        print(f"- 아이템당 평균 시간: {total_time/total_items:.2f}분")
        print("="*50)
        
        # 날짜 열을 문자열로 변환
        result_df['ds'] = result_df['ds'].dt.strftime('%Y-%m-%d')
        
        print(f"최종 결과 DataFrame 형태: {result_df.shape}")
        print(f"최종 결과 데이터 컬럼: {result_df.columns.tolist()}")
        
        return result_df

    def train_and_predict(self, args):
        item_id, train_data = args
        try:
            # Prophet 모델 학습
            model = self.train_prophet_model(train_data)
            
            # 예측 기간 설정 (2016-06-20부터 2017-12-31까지)
            future_dates = pd.date_range(start='2016-06-20', end='2017-12-31', freq='D')
            future = pd.DataFrame({'ds': future_dates})
            
            # 예측 수행
            forecast = model.predict(future)
            
            # 결과 저장
            result = forecast[['ds', 'yhat']].copy()
            result.columns = ['ds', f'F{item_id}']
            return result
        except Exception as e:
            print(f"Error processing item {item_id}: {str(e)}")
            return None

dataprocessor = DataProcessor()
dataprocessor.prepare_prophet_data()

# 모든 아이템에 대해 예측 수행
prediction_df = dataprocessor.predict_all_items()
prediction_df.to_csv("prediction_df.csv", index=True)
print("\n예측이 완료되었습니다. 결과는 prediction_df.csv 파일에 저장되었습니다.")
