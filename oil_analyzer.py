import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class OilAnalyzer:
    """
    Sistema di analisi multi-dimensionale per dati petrolio
    Ogni metrica applicabile a qualsiasi timeframe con qualsiasi combinazione di filtri
    """
    
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self._prepare_data()
        self.current_filters = {}
        self.filtered_df = self.df.copy()
        
    def _prepare_data(self):
        """Prepara e arricchisce il dataset"""
        self.df = self.original_df.copy()
        
        # Conversione date e ordinamento
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Dimensioni temporali aggiuntive
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Week_Year'] = self.df['Date'].dt.isocalendar().week
        self.df['Day_of_Month'] = self.df['Date'].dt.day
        
        # Classificazioni return (usando Change_Pct come return)
        self.df['Return'] = self.df['Change_Pct']
        self.df['Return_Positive'] = self.df['Return'] > 0
        self.df['Return_Negative'] = self.df['Return'] < 0
        self.df['Return_Neutral'] = self.df['Return'] == 0
        
        # Metriche di range aggiuntive
        if 'Daily_Range' not in self.df.columns:
            self.df['Daily_Range'] = self.df['High'] - self.df['Low']
        
        # Range percentuale (range / prezzo medio del giorno)
        midpoint = (self.df['High'] + self.df['Low']) / 2
        self.df['Range_Percentage'] = (self.df['Daily_Range'] / midpoint) * 100
        
        # Range normalizzato rispetto al close
        self.df['Range_Normalized'] = (self.df['Daily_Range'] / self.df['Close']) * 100
        
        # Usa Is_Positive se disponibile
        if 'Is_Positive' in self.df.columns:
            self.df['Return_Positive'] = self.df['Is_Positive'] == 1
        
    def apply_filters(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Applica combinazione di filtri ai dati"""
        filtered_df = self.df.copy()
        
        for filter_name, filter_value in filters.items():
            if filter_value is None or filter_value == [] or filter_value == "All":
                continue
                
            if filter_name == "years" and isinstance(filter_value, list):
                filtered_df = filtered_df[filtered_df['Year'].isin(filter_value)]
            elif filter_name == "quarters" and isinstance(filter_value, list):
                # Il dataset ha Quarter come "2010-Q1", ma il filtro usa "Q1"
                # Filtra per tutti i quarter che terminano con i valori selezionati
                quarter_filter = filtered_df['Quarter'].str.endswith(tuple(q[1:] for q in filter_value))  # Rimuove "Q" e cerca per "1", "2", etc.
                filtered_df = filtered_df[quarter_filter]
            elif filter_name == "months" and isinstance(filter_value, list):
                filtered_df = filtered_df[filtered_df['Month'].isin(filter_value)]
            elif filter_name == "days_of_week" and isinstance(filter_value, list):
                filtered_df = filtered_df[filtered_df['Day_of_Week'].isin(filter_value)]
            elif filter_name == "weeks_of_month" and isinstance(filter_value, list):
                filtered_df = filtered_df[filtered_df['Week_of_Month'].isin(filter_value)]
            elif filter_name == "return_type":
                if filter_value == "Positive Only":
                    filtered_df = filtered_df[filtered_df['Return_Positive']]
                elif filter_value == "Negative Only":
                    filtered_df = filtered_df[filtered_df['Return_Negative']]
                # "All" non filtra
                    
        return filtered_df
    
    def aggregate_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggrega dati al timeframe specificato"""
        if timeframe == "Daily":
            return df
        
        df = df.copy()
        
        if timeframe == "Weekly":
            # Aggregazione Lunedì-Venerdì per settimana solare
            grouped = df.groupby(['Year', 'Week_Year'])
            
            def agg_week(group):
                if len(group) == 0:
                    return None
                return pd.Series({
                    'Date': group['Date'].iloc[-1],  # Venerdì
                    'Open': group['Open'].iloc[0],   # Lunedì
                    'High': group['High'].max(),
                    'Low': group['Low'].min(),
                    'Close': group['Close'].iloc[-1], # Venerdì
                    'Volume': group['Volume'].sum(),
                    'Return': ((group['Close'].iloc[-1] - group['Open'].iloc[0]) / group['Open'].iloc[0]) * 100,
                    'ATR_14': group['ATR_14'].mean(),
                    'Daily_Range': group['Daily_Range'].sum(),  # Range cumulativo
                    'Weekly_Range': group['High'].max() - group['Low'].min(),  # Range della settimana
                    'Range_Percentage': group['Range_Percentage'].mean(),
                    'Range_Normalized': group['Range_Normalized'].mean(),
                    'Cumulative_Range_Days': len(group),  # Giorni nella settimana
                })
            
            result = grouped.apply(agg_week).reset_index(drop=True)
            
        elif timeframe == "Monthly":
            grouped = df.groupby(['Year', 'Month'])
            
            def agg_month(group):
                if len(group) == 0:
                    return None
                return pd.Series({
                    'Date': group['Date'].iloc[-1],
                    'Open': group['Open'].iloc[0],
                    'High': group['High'].max(),
                    'Low': group['Low'].min(),
                    'Close': group['Close'].iloc[-1],
                    'Volume': group['Volume'].sum(),
                    'Return': ((group['Close'].iloc[-1] - group['Open'].iloc[0]) / group['Open'].iloc[0]) * 100,
                    'ATR_14': group['ATR_14'].mean(),
                    'Daily_Range': group['Daily_Range'].sum(),  # Range cumulativo del mese
                    'Monthly_Range': group['High'].max() - group['Low'].min(),  # Range del mese
                    'Range_Percentage': group['Range_Percentage'].mean(),
                    'Range_Normalized': group['Range_Normalized'].mean(),
                    'Cumulative_Range_Days': len(group),  # Giorni nel mese
                })
            
            result = grouped.apply(agg_month).reset_index(drop=True)
            
        elif timeframe == "Yearly":
            grouped = df.groupby('Year')
            
            def agg_year(group):
                if len(group) == 0:
                    return None
                return pd.Series({
                    'Date': group['Date'].iloc[-1],
                    'Open': group['Open'].iloc[0],
                    'High': group['High'].max(),
                    'Low': group['Low'].min(),
                    'Close': group['Close'].iloc[-1],
                    'Volume': group['Volume'].sum(),
                    'Return': ((group['Close'].iloc[-1] - group['Open'].iloc[0]) / group['Open'].iloc[0]) * 100,
                    'ATR_14': group['ATR_14'].mean(),
                    'Daily_Range': group['Daily_Range'].sum(),  # Range cumulativo dell'anno
                    'Yearly_Range': group['High'].max() - group['Low'].min(),  # Range dell'anno
                    'Range_Percentage': group['Range_Percentage'].mean(),
                    'Range_Normalized': group['Range_Normalized'].mean(),
                    'Cumulative_Range_Days': len(group),  # Giorni nell'anno
                })
            
            result = grouped.apply(agg_year).reset_index(drop=True)
        
        # Ricalcola classificazioni return per timeframe aggregato
        result['Return_Positive'] = result['Return'] > 0
        result['Return_Negative'] = result['Return'] < 0
        result['Return_Neutral'] = result['Return'] == 0
        
        return result.dropna()
    
    def calculate_returns_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcola metriche returns"""
        if len(df) == 0:
            return {}
            
        return {
            'Average_Return': df['Return'].mean(),
            'Average_Positive_Return': df[df['Return_Positive']]['Return'].mean() if len(df[df['Return_Positive']]) > 0 else np.nan,
            'Average_Negative_Return': df[df['Return_Negative']]['Return'].mean() if len(df[df['Return_Negative']]) > 0 else np.nan,
        }
    
    def calculate_win_rate_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcola win rate metrics"""
        if len(df) == 0:
            return {}
            
        total = len(df)
        return {
            'Positive_Periods_Percentage': (df['Return_Positive'].sum() / total) * 100,
            'Negative_Periods_Percentage': (df['Return_Negative'].sum() / total) * 100,
            'Neutral_Periods_Percentage': (df['Return_Neutral'].sum() / total) * 100,
        }
    
    def calculate_streak_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcola streak analysis completa"""
        if len(df) == 0:
            return {}
        
        # Calcola streak consecutive
        df = df.copy()
        df['Sign_Change'] = (df['Return_Positive'] != df['Return_Positive'].shift(1)).fillna(True)
        df['Streak_Group'] = df['Sign_Change'].cumsum()
        
        streaks = []
        for group_id, group in df.groupby('Streak_Group'):
            if len(group) > 0:
                is_positive = group['Return_Positive'].iloc[0]
                length = len(group)
                cumulative_return = ((1 + group['Return']/100).prod() - 1) * 100
                streaks.append({
                    'length': length,
                    'is_positive': is_positive,
                    'cumulative_return': cumulative_return
                })
        
        if not streaks:
            return {}
        
        streaks_df = pd.DataFrame(streaks)
        positive_streaks = streaks_df[streaks_df['is_positive']]
        negative_streaks = streaks_df[~streaks_df['is_positive']]
        
        # A) Medie delle lunghezze
        avg_positive_length = positive_streaks['length'].mean() if len(positive_streaks) > 0 else np.nan
        avg_negative_length = negative_streaks['length'].mean() if len(negative_streaks) > 0 else np.nan
        
        # B) Distribuzione frequenze con return medi
        positive_freq = {}
        positive_freq_returns = {}
        if len(positive_streaks) > 0:
            for length in positive_streaks['length'].unique():
                length_data = positive_streaks[positive_streaks['length'] == length]
                positive_freq[length] = len(length_data)
                positive_freq_returns[length] = length_data['cumulative_return'].mean()
        
        negative_freq = {}
        negative_freq_returns = {}
        if len(negative_streaks) > 0:
            for length in negative_streaks['length'].unique():
                length_data = negative_streaks[negative_streaks['length'] == length]
                negative_freq[length] = len(length_data)
                negative_freq_returns[length] = length_data['cumulative_return'].mean()
        
        # C) Return durante streak
        avg_positive_return = positive_streaks['cumulative_return'].mean() if len(positive_streaks) > 0 else np.nan
        avg_negative_return = negative_streaks['cumulative_return'].mean() if len(negative_streaks) > 0 else np.nan
        
        return {
            'Avg_Positive_Streak_Length': avg_positive_length,
            'Avg_Negative_Streak_Length': avg_negative_length,
            'Positive_Streak_Frequencies': positive_freq,
            'Negative_Streak_Frequencies': negative_freq,
            'Positive_Streak_Returns': positive_freq_returns,
            'Negative_Streak_Returns': negative_freq_returns,
            'Avg_Positive_Streak_Return': avg_positive_return,
            'Avg_Negative_Streak_Return': avg_negative_return,
        }
    
    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcola volatility metrics"""
        if len(df) == 0 or len(df) < 2:
            return {}
        
        # Volatility clustering (autocorrelazione della volatilità)
        volatility = df['ATR_14'].fillna(0)
        high_vol_threshold = volatility.quantile(0.7)
        
        high_vol_days = volatility > high_vol_threshold
        clustering = 0
        
        if len(high_vol_days) > 1:
            # P(alta_vol_domani | alta_vol_oggi)
            high_vol_tomorrow = high_vol_days.shift(-1).fillna(False)
            clustering = (high_vol_days & high_vol_tomorrow).sum() / high_vol_days.sum() if high_vol_days.sum() > 0 else 0
        
        return {
            'ATR_Mean': df['ATR_14'].mean(),
            'Range_Mean': df['Daily_Range'].mean(),
            'Volatility_Clustering': clustering * 100,  # Percentuale
        }
    
    def calculate_range_metrics(self, df: pd.DataFrame, timeframe: str) -> Dict[str, float]:
        """Calcola metriche di range avanzate"""
        if len(df) == 0:
            return {}
        
        results = {}
        
        # Metriche base sempre disponibili
        results['Range_Mean_Absolute'] = df['Daily_Range'].mean()
        
        # Range percentuale (elimino Range_Normalized - ridondante)
        if 'Range_Percentage' in df.columns:
            results['Range_Mean_Percentage'] = df['Range_Percentage'].mean()
        
        # Metriche specifiche per timeframe aggregati
        if timeframe == "Weekly" and 'Weekly_Range' in df.columns:
            results['Weekly_Range_Mean'] = df['Weekly_Range'].mean()
            results['Cumulative_vs_Period_Range_Ratio'] = df['Daily_Range'].mean() / df['Weekly_Range'].mean() if df['Weekly_Range'].mean() > 0 else np.nan
            # Range medio per giorno in settimane
            if 'Cumulative_Range_Days' in df.columns:
                results['Daily_Range_Per_Week_Day'] = df['Daily_Range'].sum() / df['Cumulative_Range_Days'].sum() if df['Cumulative_Range_Days'].sum() > 0 else np.nan
                
        elif timeframe == "Monthly" and 'Monthly_Range' in df.columns:
            results['Monthly_Range_Mean'] = df['Monthly_Range'].mean()
            results['Cumulative_vs_Period_Range_Ratio'] = df['Daily_Range'].mean() / df['Monthly_Range'].mean() if df['Monthly_Range'].mean() > 0 else np.nan
            # Range medio per giorno in mesi
            if 'Cumulative_Range_Days' in df.columns:
                results['Daily_Range_Per_Month_Day'] = df['Daily_Range'].sum() / df['Cumulative_Range_Days'].sum() if df['Cumulative_Range_Days'].sum() > 0 else np.nan
                
        elif timeframe == "Yearly" and 'Yearly_Range' in df.columns:
            results['Yearly_Range_Mean'] = df['Yearly_Range'].mean()
            results['Cumulative_vs_Period_Range_Ratio'] = df['Daily_Range'].mean() / df['Yearly_Range'].mean() if df['Yearly_Range'].mean() > 0 else np.nan
            # Range medio per giorno in anni
            if 'Cumulative_Range_Days' in df.columns:
                results['Daily_Range_Per_Year_Day'] = df['Daily_Range'].sum() / df['Cumulative_Range_Days'].sum() if df['Cumulative_Range_Days'].sum() > 0 else np.nan
        
        # Range volatility (volatilità del range stesso)
        range_volatility = df['Daily_Range'].std()
        results['Range_Volatility'] = range_volatility
        
        # Classificazione Range Volatility per context
        mean_range = df['Daily_Range'].mean()
        if mean_range > 0:
            volatility_ratio = range_volatility / mean_range
            if volatility_ratio > 0.8:
                results['Range_Volatility_Level'] = "High"
            elif volatility_ratio > 0.5:
                results['Range_Volatility_Level'] = "Medium"
            else:
                results['Range_Volatility_Level'] = "Low"
        else:
            results['Range_Volatility_Level'] = "N/A"
        
        # Range efficiency (quanto del range viene "utilizzato" dai movimenti di prezzo)
        if len(df) > 1:
            price_moves = abs(df['Close'] - df['Open'])
            results['Range_Efficiency'] = (price_moves / df['Daily_Range']).mean() * 100
            
        # High/Low range distribution
        if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
            # Dove chiude rispetto al range (0 = low, 1 = high)
            valid_range_mask = df['Daily_Range'] > 0  # Solo giorni con range > 0
            valid_df = df[valid_range_mask]
            
            if len(valid_df) > 0:
                range_position = (valid_df['Close'] - valid_df['Low']) / valid_df['Daily_Range']
                
                results['Avg_Close_Range_Position'] = range_position.mean() * 100  # Percentuale
                results['Range_Close_Near_High_Pct'] = (range_position > 0.8).sum() / len(valid_df) * 100
                results['Range_Close_Near_Low_Pct'] = (range_position < 0.2).sum() / len(valid_df) * 100
                
                # Bias direction indicator
                if range_position.mean() > 0.6:
                    results['Range_Position_Bias'] = "Bullish"
                elif range_position.mean() < 0.4:
                    results['Range_Position_Bias'] = "Bearish" 
                else:
                    results['Range_Position_Bias'] = "Neutral"
            else:
                results['Avg_Close_Range_Position'] = np.nan
                results['Range_Close_Near_High_Pct'] = np.nan
                results['Range_Close_Near_Low_Pct'] = np.nan
                results['Range_Position_Bias'] = "N/A"
        
        return results
    
    def calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcola volume metrics foundation"""
        if len(df) == 0:
            return {}
        
        volume_mean = df['Volume'].mean()
        above_avg_vol = (df['Volume'] > volume_mean).sum() / len(df) * 100
        
        # Correlazione volume con return type
        pos_vol = df[df['Return_Positive']]['Volume'].mean() if len(df[df['Return_Positive']]) > 0 else np.nan
        neg_vol = df[df['Return_Negative']]['Volume'].mean() if len(df[df['Return_Negative']]) > 0 else np.nan
        
        return {
            'Volume_Mean': volume_mean,
            'Days_Above_Avg_Volume_Pct': above_avg_vol,
            'Positive_Days_Avg_Volume': pos_vol,
            'Negative_Days_Avg_Volume': neg_vol,
        }
    
    def calculate_best_worst_periods(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcola best/worst periods"""
        if len(df) == 0:
            return {}
        
        return {
            'Best_Single_Period': df['Return'].max(),
            'Worst_Single_Period': df['Return'].min(),
        }
    
    def calculate_extreme_events(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcola extreme events con soglie percentuali (1%, 3%, 5%)"""
        if len(df) == 0:
            return {}
        
        returns = df['Return']
        
        # Soglie percentuali fisse
        thresholds = [1, 3, 5]
        results = {}
        
        for threshold in thresholds:
            # Movimenti positivi sopra soglia
            days_above = (returns > threshold).sum()
            days_above_pct = (days_above / len(df)) * 100
            
            # Movimenti negativi sotto soglia
            days_below = (returns < -threshold).sum()
            days_below_pct = (days_below / len(df)) * 100
            
            results[f'Days_Above_{threshold}pct'] = days_above
            results[f'Days_Below_{threshold}pct'] = days_below
            results[f'Days_Above_{threshold}pct_Percentage'] = days_above_pct
            results[f'Days_Below_{threshold}pct_Percentage'] = days_below_pct
        
        # Aggiungi anche la deviazione standard per riferimento
        results['Return_Std_Dev'] = returns.std()
        
        return results
    
    def calculate_mean_reversion_advanced(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcola mean reversion metrics avanzate"""
        if len(df) == 0 or 'MA7' not in df.columns:
            return {}
        
        # Calcola streak come prima
        df_work = df.copy()
        df_work['Sign_Change'] = (df_work['Return_Positive'] != df_work['Return_Positive'].shift(1)).fillna(True)
        df_work['Streak_Group'] = df_work['Sign_Change'].cumsum()
        
        reversion_speeds_ma9 = []
        reversion_speeds_recovery = []
        overshoots = []
        best_streak_return = -np.inf
        worst_streak_return = np.inf
        
        for group_id, group in df_work.groupby('Streak_Group'):
            if len(group) == 0:
                continue
                
            is_positive = group['Return_Positive'].iloc[0]
            cumulative_return = ((1 + group['Return']/100).prod() - 1) * 100
            
            # Update best/worst streak
            if is_positive and cumulative_return > best_streak_return:
                best_streak_return = cumulative_return
            elif not is_positive and cumulative_return < worst_streak_return:
                worst_streak_return = cumulative_return
            
            # Solo per streak negative, calcola reversion
            if not is_positive and len(group) >= 2:
                streak_end_idx = group.index[-1]
                
                # Trova dati dopo la streak per calcolare recovery
                remaining_data = df_work.loc[streak_end_idx+1:]
                
                if len(remaining_data) > 0:
                    # B) Periodi per tornare sopra MA9 (usando MA7 come proxy per MA9)
                    ma9_at_end = group['MA7'].iloc[-1] if 'MA7' in group.columns else None
                    if ma9_at_end is not None:
                        for i, (idx, row) in enumerate(remaining_data.iterrows()):
                            if row['Close'] > ma9_at_end:
                                reversion_speeds_ma9.append(i + 1)
                                
                                # Calcola overshoot rispetto a MA9
                                overshoot = ((row['Close'] - ma9_at_end) / ma9_at_end) * 100
                                overshoots.append(overshoot)
                                break
                    
                    # C) Periodi per recovery matematico
                    cumulative_loss = cumulative_return / 100  # Conversione a decimale
                    if cumulative_loss < 0:  # È una perdita
                        recovery_needed = -cumulative_loss / (1 + cumulative_loss)
                        price_at_end = group['Close'].iloc[-1]
                        target_price = price_at_end * (1 + recovery_needed)
                        
                        for i, (idx, row) in enumerate(remaining_data.iterrows()):
                            if row['Close'] >= target_price:
                                reversion_speeds_recovery.append(i + 1)
                                break
        
        results = {}
        
        # Best/Worst streak returns
        if best_streak_return != -np.inf:
            results['Best_Streak_Return'] = best_streak_return
        if worst_streak_return != np.inf:
            results['Worst_Streak_Return'] = worst_streak_return
            
        # Mean reversion speeds
        if reversion_speeds_ma9:
            results['Avg_Periods_To_MA9'] = np.mean(reversion_speeds_ma9)
        if reversion_speeds_recovery:
            results['Avg_Periods_To_Recovery'] = np.mean(reversion_speeds_recovery)
            
        # Overshoot analysis
        if overshoots:
            results['Avg_Overshoot_MA9'] = np.mean(overshoots)
            
        return results
    
    def calculate_autocorrelation(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> Dict[str, float]:
        """Calcola autocorrelazione per diversi lag"""
        if len(df) == 0:
            return {}
        
        returns = df['Return'].dropna()
        correlations = {}
        
        for lag in lags:
            if len(returns) > lag:
                corr = returns.corr(returns.shift(lag))
                correlations[f'Autocorr_t-{lag}'] = corr
            else:
                correlations[f'Autocorr_t-{lag}'] = np.nan
                
        return correlations
    
    def calculate_price_action_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analizza pattern di price action avanzati"""
        if len(df) < 5:
            return {}
        
        results = {}
        
        # Doji patterns (Open ≈ Close)
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            body_size = abs(df['Close'] - df['Open'])
            total_range = df['High'] - df['Low']
            
            # Doji: body < 10% del range totale
            valid_range = total_range > 0
            doji_condition = valid_range & ((body_size / total_range) < 0.1)
            
            results['PriceAction_Doji_Days'] = doji_condition.sum()
            results['PriceAction_Doji_Percentage'] = (doji_condition.sum() / len(df)) * 100
            
            # Hammer/Shooting Star patterns
            upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
            lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
            
            # Hammer: lower shadow > 2x body size, upper shadow < body size
            hammer_condition = valid_range & (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
            results['PriceAction_Hammer_Days'] = hammer_condition.sum()
            results['PriceAction_Hammer_Percentage'] = (hammer_condition.sum() / len(df)) * 100
            
            # Shooting Star: upper shadow > 2x body size, lower shadow < body size  
            shooting_star_condition = valid_range & (upper_shadow > 2 * body_size) & (lower_shadow < body_size)
            results['PriceAction_ShootingStar_Days'] = shooting_star_condition.sum()
            results['PriceAction_ShootingStar_Percentage'] = (shooting_star_condition.sum() / len(df)) * 100
            
            # Gap analysis
            if len(df) > 1:
                prev_close = df['Close'].shift(1)
                current_open = df['Open']
                
                # Gap up: today's open > yesterday's close
                gap_up = current_open > prev_close
                gap_up_pct = ((current_open - prev_close) / prev_close * 100).fillna(0)
                
                # Gap down: today's open < yesterday's close  
                gap_down = current_open < prev_close
                gap_down_pct = ((prev_close - current_open) / prev_close * 100).fillna(0)
                
                results['PriceAction_Gap_Up_Days'] = gap_up.sum()
                results['PriceAction_Gap_Down_Days'] = gap_down.sum()
                results['PriceAction_Gap_Up_Avg_Size'] = gap_up_pct[gap_up].mean() if gap_up.any() else 0
                results['PriceAction_Gap_Down_Avg_Size'] = gap_down_pct[gap_down].mean() if gap_down.any() else 0
                
                # Fill rate per gap
                gap_up_filled = 0
                gap_down_filled = 0
                
                for i in range(1, len(df)):
                    if gap_up.iloc[i]:
                        # Gap up filled se il low di oggi tocca o va sotto il close di ieri
                        if df['Low'].iloc[i] <= prev_close.iloc[i]:
                            gap_up_filled += 1
                    
                    if gap_down.iloc[i]:
                        # Gap down filled se il high di oggi tocca o va sopra il close di ieri
                        if df['High'].iloc[i] >= prev_close.iloc[i]:
                            gap_down_filled += 1
                
                if gap_up.sum() > 0:
                    results['PriceAction_Gap_Up_Fill_Rate'] = (gap_up_filled / gap_up.sum()) * 100
                if gap_down.sum() > 0:
                    results['PriceAction_Gap_Down_Fill_Rate'] = (gap_down_filled / gap_down.sum()) * 100
        
        # Inside/Outside days
        if len(df) > 1:
            prev_high = df['High'].shift(1)
            prev_low = df['Low'].shift(1)
            
            # Inside day: today's range completamente dentro quello di ieri
            inside_days = (df['High'] < prev_high) & (df['Low'] > prev_low)
            results['PriceAction_Inside_Days'] = inside_days.sum()
            results['PriceAction_Inside_Days_Percentage'] = (inside_days.sum() / (len(df) - 1)) * 100
            
            # Outside day: today's range comprende completamente quello di ieri
            outside_days = (df['High'] > prev_high) & (df['Low'] < prev_low)
            results['PriceAction_Outside_Days'] = outside_days.sum()
            results['PriceAction_Outside_Days_Percentage'] = (outside_days.sum() / (len(df) - 1)) * 100
        
        # Strong vs weak closes
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            range_position = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            range_position = range_position.fillna(0.5)  # Default to middle for zero-range days
            
            # Strong close: close in top 20% of range
            strong_closes = range_position > 0.8
            results['PriceAction_Strong_Close_Days'] = strong_closes.sum()
            results['PriceAction_Strong_Close_Percentage'] = (strong_closes.sum() / len(df)) * 100
            
            # Weak close: close in bottom 20% of range
            weak_closes = range_position < 0.2
            results['PriceAction_Weak_Close_Days'] = weak_closes.sum()
            results['PriceAction_Weak_Close_Percentage'] = (weak_closes.sum() / len(df)) * 100
            
            # Analisi efficacia di strong/weak closes
            if len(df) > 1:
                next_day_positive = df['Return_Positive'].shift(-1)
                
                strong_close_success = next_day_positive[strong_closes].mean() * 100 if strong_closes.any() else np.nan
                weak_close_success = (1 - next_day_positive[weak_closes]).mean() * 100 if weak_closes.any() else np.nan
                
                results['PriceAction_Strong_Close_Next_Day_Success'] = strong_close_success
                results['PriceAction_Weak_Close_Next_Day_Success'] = weak_close_success
        
        return results
    
    def calculate_all_metrics(self, timeframe: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Calcola tutte le metriche per timeframe e filtri specificati"""
        # Applica filtri
        filtered_df = self.apply_filters(filters)
        
        # Aggrega al timeframe
        aggregated_df = self.aggregate_to_timeframe(filtered_df, timeframe)
        
        if len(aggregated_df) == 0:
            return {"error": "No data available for the selected filters"}
        
        # Calcola tutte le metriche
        results = {
            'timeframe': timeframe,
            'filters_applied': filters,
            'data_points': len(aggregated_df),
            'date_range': f"{aggregated_df['Date'].min().strftime('%Y-%m-%d')} to {aggregated_df['Date'].max().strftime('%Y-%m-%d')}",
        }
        
        # Returns Analysis
        returns_metrics = self.calculate_returns_analysis(aggregated_df)
        results.update({f"Returns_{k}": v for k, v in returns_metrics.items()})
        
        # Win Rate Analysis  
        winrate_metrics = self.calculate_win_rate_analysis(aggregated_df)
        results.update({f"WinRate_{k}": v for k, v in winrate_metrics.items()})
        
        # Streak Analysis
        streak_metrics = self.calculate_streak_analysis(aggregated_df)
        results.update({f"Streak_{k}": v for k, v in streak_metrics.items()})
        
        # Volatility Metrics
        vol_metrics = self.calculate_volatility_metrics(aggregated_df)
        results.update({f"Volatility_{k}": v for k, v in vol_metrics.items()})
        
        # Range Metrics (nuove metriche avanzate)
        range_metrics = self.calculate_range_metrics(aggregated_df, timeframe)
        results.update({f"Range_{k}": v for k, v in range_metrics.items()})
        
        # Volume Metrics
        volume_metrics = self.calculate_volume_metrics(aggregated_df)
        results.update({f"Volume_{k}": v for k, v in volume_metrics.items()})
        
        # Best/Worst Periods
        best_worst_metrics = self.calculate_best_worst_periods(aggregated_df)
        results.update({f"BestWorst_{k}": v for k, v in best_worst_metrics.items()})
        
        # Extreme Events
        extreme_metrics = self.calculate_extreme_events(aggregated_df)
        results.update({f"Extreme_{k}": v for k, v in extreme_metrics.items()})
        
        # Mean Reversion Advanced
        mean_reversion_advanced = self.calculate_mean_reversion_advanced(aggregated_df)
        results.update({f"MeanRev_{k}": v for k, v in mean_reversion_advanced.items()})
        
        # Autocorrelation
        autocorr_metrics = self.calculate_autocorrelation(aggregated_df)
        results.update(autocorr_metrics)
        
        # Price Action Patterns
        price_action_metrics = self.calculate_price_action_patterns(aggregated_df)
        results.update({f"PriceAction_{k}": v for k, v in price_action_metrics.items()})
        
        # Weekday Behavior Analysis (solo per timeframe Daily)
        if timeframe == "Daily":
            weekday_metrics = self.calculate_weekday_behavior_analysis(aggregated_df)
            results.update({f"Weekday_{k}": v for k, v in weekday_metrics.items()})
        
        return results

    def calculate_weekday_behavior_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizza le probabilità condizionali tra comportamento del lunedì e performance settimanale.
        
        Settimana positiva = Close Venerdì > Open Lunedì
        Lunedì positivo = Return del lunedì > 0
        
        Calcola le 4 probabilità condizionali:
        - P(settimana positiva | lunedì positivo)
        - P(settimana negativa | lunedì positivo) 
        - P(settimana positiva | lunedì negativo)
        - P(settimana negativa | lunedì negativo)
        """
        if len(df) == 0 or 'Day_of_Week' not in df.columns:
            return {}
        
        results = {}
        
        # Verifica formato giorni (italiano vs inglese)
        sample_days = df['Day_of_Week'].dropna().unique()
        use_italian = any(day in sample_days for day in ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì'])
        
        monday_name = 'Lunedì' if use_italian else 'Monday'
        friday_name = 'Venerdì' if use_italian else 'Friday'
        
        # Assicurati che il DataFrame sia ordinato per data
        df_sorted = df.sort_values('Date').reset_index(drop=True)
        
        # Estrai Year e Week_Year dalla colonna Date se non presenti
        if 'Year' not in df_sorted.columns:
            df_sorted['Year'] = pd.to_datetime(df_sorted['Date']).dt.year
        if 'Week_Year' not in df_sorted.columns:
            df_sorted['Week_Year'] = pd.to_datetime(df_sorted['Date']).dt.isocalendar().week
        
        # Usa Change_Pct come Return se Return non è disponibile
        if 'Return' not in df_sorted.columns and 'Change_Pct' in df_sorted.columns:
            df_sorted['Return'] = df_sorted['Change_Pct']
        
        # Raggruppa per settimane (Year, Week_Year)
        weekly_analysis = []
        
        for (year, week), week_group in df_sorted.groupby(['Year', 'Week_Year']):
            if len(week_group) < 2:  # Serve almeno lunedì E venerdì (o altri giorni)
                continue
                
            # Trova lunedì e venerdì
            monday_data = week_group[week_group['Day_of_Week'] == monday_name]
            friday_data = week_group[week_group['Day_of_Week'] == friday_name]
            
            if len(monday_data) > 0 and len(friday_data) > 0:
                # Dati del lunedì
                monday_return = monday_data['Return'].iloc[0]
                monday_open = monday_data['Open'].iloc[0]
                monday_positive = monday_return > 0
                
                # Dati del venerdì  
                friday_close = friday_data['Close'].iloc[0]
                
                # Performance settimanale: Close Venerdì vs Open Lunedì
                weekly_return_pct = ((friday_close - monday_open) / monday_open) * 100
                weekly_positive = weekly_return_pct > 0
                
                weekly_analysis.append({
                    'year': year,
                    'week': week,
                    'monday_return': monday_return,
                    'monday_positive': monday_positive,
                    'weekly_return_pct': weekly_return_pct,
                    'weekly_positive': weekly_positive,
                    'monday_open': monday_open,
                    'friday_close': friday_close
                })
        
        if not weekly_analysis:
            return {'error': 'Nessun dato sufficiente per analisi settimanale (serve almeno lunedì + venerdì)'}
        
        # Converti in DataFrame per analisi
        weekly_df = pd.DataFrame(weekly_analysis)
        
        # Calcola le 4 probabilità condizionali
        total_weeks = len(weekly_df)
        
        # Filtra per lunedì positivi e negativi
        monday_positive_weeks = weekly_df[weekly_df['monday_positive']]
        monday_negative_weeks = weekly_df[~weekly_df['monday_positive']]
        
        # 1. P(settimana positiva | lunedì positivo)
        if len(monday_positive_weeks) > 0:
            week_pos_given_monday_pos = (monday_positive_weeks['weekly_positive'].sum() / len(monday_positive_weeks)) * 100
            results['Prob_Week_Positive_Given_Monday_Positive'] = week_pos_given_monday_pos
        else:
            results['Prob_Week_Positive_Given_Monday_Positive'] = np.nan
            
        # 2. P(settimana negativa | lunedì positivo)  
        if len(monday_positive_weeks) > 0:
            week_neg_given_monday_pos = ((len(monday_positive_weeks) - monday_positive_weeks['weekly_positive'].sum()) / len(monday_positive_weeks)) * 100
            results['Prob_Week_Negative_Given_Monday_Positive'] = week_neg_given_monday_pos
        else:
            results['Prob_Week_Negative_Given_Monday_Positive'] = np.nan
            
        # 3. P(settimana positiva | lunedì negativo)
        if len(monday_negative_weeks) > 0:
            week_pos_given_monday_neg = (monday_negative_weeks['weekly_positive'].sum() / len(monday_negative_weeks)) * 100
            results['Prob_Week_Positive_Given_Monday_Negative'] = week_pos_given_monday_neg
        else:
            results['Prob_Week_Positive_Given_Monday_Negative'] = np.nan
            
        # 4. P(settimana negativa | lunedì negativo)
        if len(monday_negative_weeks) > 0:
            week_neg_given_monday_neg = ((len(monday_negative_weeks) - monday_negative_weeks['weekly_positive'].sum()) / len(monday_negative_weeks)) * 100
            results['Prob_Week_Negative_Given_Monday_Negative'] = week_neg_given_monday_neg
        else:
            results['Prob_Week_Negative_Given_Monday_Negative'] = np.nan
        
        # Statistiche aggiuntive utili
        results['Total_Weeks_Analyzed'] = total_weeks
        results['Monday_Positive_Weeks_Count'] = len(monday_positive_weeks)
        results['Monday_Negative_Weeks_Count'] = len(monday_negative_weeks)
        results['Overall_Weekly_Positive_Rate'] = (weekly_df['weekly_positive'].sum() / total_weeks) * 100
        results['Overall_Monday_Positive_Rate'] = (weekly_df['monday_positive'].sum() / total_weeks) * 100
        
        # Correlazione tra return del lunedì e return settimanale
        monday_weekly_correlation = weekly_df['monday_return'].corr(weekly_df['weekly_return_pct'])
        results['Monday_Weekly_Return_Correlation'] = monday_weekly_correlation
        
        # Return medi
        results['Avg_Weekly_Return_When_Monday_Positive'] = monday_positive_weeks['weekly_return_pct'].mean() if len(monday_positive_weeks) > 0 else np.nan
        results['Avg_Weekly_Return_When_Monday_Negative'] = monday_negative_weeks['weekly_return_pct'].mean() if len(monday_negative_weeks) > 0 else np.nan
        
        # ===== NUOVE ANALISI =====
        
        # 1. PROBABILITÀ LUNEDÌ → ALTRI GIORNI DELLA SETTIMANA
        tuesday_name = 'Martedì' if use_italian else 'Tuesday'
        wednesday_name = 'Mercoledì' if use_italian else 'Wednesday'
        thursday_name = 'Giovedì' if use_italian else 'Thursday'
        
        other_days = [tuesday_name, wednesday_name, thursday_name, friday_name]
        english_other_days = ['Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        for i, day_name in enumerate(other_days):
            english_day = english_other_days[i]
            
            # Trova coppie Lunedì → giorno_x nella stessa settimana
            monday_to_day_analysis = []
            
            for (year, week), week_group in df_sorted.groupby(['Year', 'Week_Year']):
                monday_data = week_group[week_group['Day_of_Week'] == monday_name]
                day_data = week_group[week_group['Day_of_Week'] == day_name]
                
                if len(monday_data) > 0 and len(day_data) > 0:
                    monday_return = monday_data['Return'].iloc[0]
                    day_return = day_data['Return'].iloc[0]
                    monday_positive = monday_return > 0
                    day_positive = day_return > 0
                    
                    monday_to_day_analysis.append({
                        'monday_positive': monday_positive,
                        'day_positive': day_positive
                    })
            
            if monday_to_day_analysis:
                monday_to_day_df = pd.DataFrame(monday_to_day_analysis)
                
                # Filtra per lunedì positivi e negativi
                monday_pos = monday_to_day_df[monday_to_day_df['monday_positive']]
                monday_neg = monday_to_day_df[~monday_to_day_df['monday_positive']]
                
                # P(giorno_x positivo | lunedì positivo)
                if len(monday_pos) > 0:
                    results[f'Prob_{english_day}_Positive_Given_Monday_Positive'] = (monday_pos['day_positive'].sum() / len(monday_pos)) * 100
                    results[f'Prob_{english_day}_Negative_Given_Monday_Positive'] = ((len(monday_pos) - monday_pos['day_positive'].sum()) / len(monday_pos)) * 100
                else:
                    results[f'Prob_{english_day}_Positive_Given_Monday_Positive'] = np.nan
                    results[f'Prob_{english_day}_Negative_Given_Monday_Positive'] = np.nan
                
                # P(giorno_x positivo | lunedì negativo)
                if len(monday_neg) > 0:
                    results[f'Prob_{english_day}_Positive_Given_Monday_Negative'] = (monday_neg['day_positive'].sum() / len(monday_neg)) * 100
                    results[f'Prob_{english_day}_Negative_Given_Monday_Negative'] = ((len(monday_neg) - monday_neg['day_positive'].sum()) / len(monday_neg)) * 100
                else:
                    results[f'Prob_{english_day}_Positive_Given_Monday_Negative'] = np.nan
                    results[f'Prob_{english_day}_Negative_Given_Monday_Negative'] = np.nan
                
                # Conteggi per statistica
                results[f'{english_day}_Monday_Pairs_Analyzed'] = len(monday_to_day_analysis)
        
        # 2. PROBABILITÀ VENERDÌ PRECEDENTE → LUNEDÌ SUCCESSIVO
        friday_to_monday_analysis = []
        
        # Raggruppa per trovare coppie Venerdì-Lunedì consecutive
        df_sorted_by_date = df_sorted.sort_values('Date').reset_index(drop=True)
        
        for i in range(len(df_sorted_by_date) - 1):
            current_row = df_sorted_by_date.iloc[i]
            next_row = df_sorted_by_date.iloc[i + 1]
            
            current_date = pd.to_datetime(current_row['Date'])
            next_date = pd.to_datetime(next_row['Date'])
            
            # Verifica se è Venerdì seguito da Lunedì (max 3 giorni di distanza per gestire weekend)
            if (current_row['Day_of_Week'] == friday_name and 
                next_row['Day_of_Week'] == monday_name and 
                (next_date - current_date).days <= 3):
                
                friday_return = current_row['Return']
                monday_return = next_row['Return']
                friday_positive = friday_return > 0
                monday_positive = monday_return > 0
                
                friday_to_monday_analysis.append({
                    'friday_positive': friday_positive,
                    'monday_positive': monday_positive,
                    'friday_date': current_date,
                    'monday_date': next_date
                })
        
        if friday_to_monday_analysis:
            friday_to_monday_df = pd.DataFrame(friday_to_monday_analysis)
            
            # Filtra per venerdì positivi e negativi
            friday_pos = friday_to_monday_df[friday_to_monday_df['friday_positive']]
            friday_neg = friday_to_monday_df[~friday_to_monday_df['friday_positive']]
            
            # P(lunedì positivo | venerdì precedente positivo)
            if len(friday_pos) > 0:
                results['Prob_Monday_Positive_Given_Friday_Previous_Positive'] = (friday_pos['monday_positive'].sum() / len(friday_pos)) * 100
                results['Prob_Monday_Negative_Given_Friday_Previous_Positive'] = ((len(friday_pos) - friday_pos['monday_positive'].sum()) / len(friday_pos)) * 100
            else:
                results['Prob_Monday_Positive_Given_Friday_Previous_Positive'] = np.nan
                results['Prob_Monday_Negative_Given_Friday_Previous_Positive'] = np.nan
            
            # P(lunedì positivo | venerdì precedente negativo)
            if len(friday_neg) > 0:
                results['Prob_Monday_Positive_Given_Friday_Previous_Negative'] = (friday_neg['monday_positive'].sum() / len(friday_neg)) * 100
                results['Prob_Monday_Negative_Given_Friday_Previous_Negative'] = ((len(friday_neg) - friday_neg['monday_positive'].sum()) / len(friday_neg)) * 100
            else:
                results['Prob_Monday_Positive_Given_Friday_Previous_Negative'] = np.nan
                results['Prob_Monday_Negative_Given_Friday_Previous_Negative'] = np.nan
            
            # Statistiche aggiuntive
            results['Friday_Monday_Consecutive_Pairs_Analyzed'] = len(friday_to_monday_analysis)
            results['Friday_Previous_Positive_Count'] = len(friday_pos)
            results['Friday_Previous_Negative_Count'] = len(friday_neg)
        
        return results
    
def create_streamlit_app():
    """Crea l'interfaccia Streamlit"""
    st.set_page_config(
        page_title="Oil Market Multi-Dimensional Analyzer",
        page_icon="🛢️",
        layout="wide"
    )
    
    st.title("🛢️ Oil Market Multi-Dimensional Analyzer")
    st.markdown("**Analisi componibile**: Ogni metrica applicabile a qualsiasi timeframe con qualsiasi combinazione di filtri")
    
    # Opzione per usare dati pre-caricati o caricare file
    data_source = st.radio(
        "📂 Sorgente Dati",
        ["Usa Dataset Pre-caricato", "Carica Nuovo File CSV"],
        help="Scegli se usare il dataset oil già incluso o caricare un nuovo file"
    )
    
    df = None
    
    if data_source == "Usa Dataset Pre-caricato":
        # Carica il dataset pre-incluso
        try:
            # Prova prima il path assoluto, poi quello relativo
            try:
                df = pd.read_csv('/Users/e/Desktop/oilProva/oil_processed.csv')
            except FileNotFoundError:
                df = pd.read_csv('oil_processed.csv')
            
            st.success(f"✅ Dataset pre-caricato: {len(df)} righe, {len(df.columns)} colonne")
            st.markdown(f"**Periodo**: {df['Date'].min()} - {df['Date'].max()}")
            st.info("💡 Usando il dataset oil completo già processato (2010-2025)")
        except Exception as e:
            st.error(f"❌ Errore nel caricamento del dataset pre-incluso: {str(e)}")
            st.warning("Prova a caricare un file CSV manualmente")
    
    else:  # Carica Nuovo File CSV
        # Caricamento dati
        uploaded_file = st.file_uploader("Carica file CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Dataset caricato: {len(df)} righe, {len(df.columns)} colonne")
                st.markdown(f"**Periodo**: {df['Date'].min()} - {df['Date'].max()}")
            except Exception as e:
                st.error(f"❌ Errore nel caricamento del file: {str(e)}")
                st.info("Assicurati che il CSV contenga le colonne richieste")
    
    # Processa i dati se disponibili
    if df is not None:
        try:
            analyzer = OilAnalyzer(df)
            
            # Sidebar con filtri
            st.sidebar.header("🔧 Filtri Combinabili")
            
            # Reset button
            if st.sidebar.button("🔄 Reset Tutti i Filtri"):
                st.experimental_rerun()
            
            # Timeframe selection
            timeframe = st.sidebar.selectbox(
                "📊 Timeframe",
                ["Daily", "Weekly", "Monthly", "Yearly"],
                help="Seleziona il timeframe per l'aggregazione"
            )
            
            # Filtri
            filters = {}
            
            # Years filter
            if 'Year' in df.columns:
                available_years = sorted(df['Year'].unique())
            else:
                # Converti Date se necessario
                date_col = pd.to_datetime(df['Date'])
                available_years = sorted(date_col.dt.year.unique())
            
            selected_years = st.sidebar.multiselect(
                "📅 Anni",
                available_years,
                help="Seleziona anni specifici"
            )
            filters['years'] = selected_years if selected_years else None
            
            # Quarters filter
            if 'Quarter' in df.columns:
                # Estrae solo la parte Q1, Q2, Q3, Q4 dai quarter completi (es: "2010-Q1" -> "Q1")
                quarter_parts = df['Quarter'].str.split('-Q').str[1]  # Prende la parte dopo "-Q"
                available_quarters = sorted(['Q' + q for q in quarter_parts.dropna().unique() if q.isdigit()])
            else:
                available_quarters = ['Q1', 'Q2', 'Q3', 'Q4']
            
            selected_quarters = st.sidebar.multiselect(
                "📊 Quarter",
                available_quarters,
                help="Seleziona quarter specifici"
            )
            filters['quarters'] = selected_quarters if selected_quarters else None
            
            # Months filter
            if 'Month' in df.columns:
                available_months = sorted(df['Month'].unique())
            else:
                available_months = list(range(1, 13))
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            # Crea mapping mese numero -> nome
            month_display = [month_names[m-1] for m in available_months]
            selected_month_names = st.sidebar.multiselect(
                "📅 Mesi",
                month_display,
                help="Seleziona mesi specifici"
            )
            # Converti indietro a numeri
            month_name_to_num = {month_names[i]: i+1 for i in range(12)}
            selected_months = [month_name_to_num[name] for name in selected_month_names] if selected_month_names else None
            filters['months'] = selected_months
            
            # Days of week filter
            if 'Day_of_Week' in df.columns:
                available_days = sorted(df['Day_of_Week'].unique())
            else:
                available_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            selected_days = st.sidebar.multiselect(
                "📅 Giorni Settimana",
                available_days,
                help="Seleziona giorni specifici della settimana"
            )
            filters['days_of_week'] = selected_days if selected_days else None
            
            # Weeks of month filter
            if 'Week_of_Month' in df.columns:
                available_weeks = sorted(df['Week_of_Month'].unique())
            else:
                available_weeks = [1, 2, 3, 4, 5]
            selected_weeks = st.sidebar.multiselect(
                "📅 Settimane del Mese",
                available_weeks,
                help="Seleziona settimane specifiche del mese"
            )
            filters['weeks_of_month'] = selected_weeks if selected_weeks else None
            
            # Return type filter
            return_type = st.sidebar.radio(
                "📈 Tipo Periodo",
                ["All", "Positive Only", "Negative Only"],
                help="Filtra per tipo di return"
            )
            filters['return_type'] = return_type
            
            # Calcola metriche
            with st.spinner("Calcolando metriche..."):
                results = analyzer.calculate_all_metrics(timeframe, filters)
            
            if "error" in results:
                st.error(results["error"])
            else:
                # Display results
                st.header("📊 Risultati Analisi")
                
                # Summary info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Timeframe", results['timeframe'])
                with col2:
                    st.metric("Data Points", results['data_points'])
                with col3:
                    st.metric("Date Range", results['date_range'])
                with col4:
                    active_filters = len([v for v in filters.values() if v not in [None, [], "All"]])
                    st.metric("Active Filters", active_filters)
                
                # Organize metrics in sections
                st.subheader("💰 Returns Analysis")
                returns_cols = st.columns(3)
                metrics_map = {
                    'Returns_Average_Return': 'Average Return (%)',
                    'Returns_Average_Positive_Return': 'Avg Positive Return (%)',
                    'Returns_Average_Negative_Return': 'Avg Negative Return (%)',
                }
                for i, (key, label) in enumerate(metrics_map.items()):
                    if key in results:
                        with returns_cols[i % 3]:
                            st.metric(label, f"{results[key]:.2f}" if not pd.isna(results[key]) else "N/A")
                
                st.subheader("🎯 Win Rate Analysis")
                winrate_cols = st.columns(3)
                winrate_map = {
                    'WinRate_Positive_Periods_Percentage': 'Positive Periods (%)',
                    'WinRate_Negative_Periods_Percentage': 'Negative Periods (%)', 
                    'WinRate_Neutral_Periods_Percentage': 'Neutral Periods (%)',
                }
                for i, (key, label) in enumerate(winrate_map.items()):
                    if key in results:
                        with winrate_cols[i % 3]:
                            st.metric(label, f"{results[key]:.1f}" if not pd.isna(results[key]) else "N/A")
                
                st.subheader("🔄 Streak Analysis")
                streak_cols = st.columns(4)
                streak_map = {
                    'Streak_Avg_Positive_Streak_Length': 'Avg Positive Streak',
                    'Streak_Avg_Negative_Streak_Length': 'Avg Negative Streak',
                    'Streak_Avg_Positive_Streak_Return': 'Avg Positive Streak Return (%)',
                    'Streak_Avg_Negative_Streak_Return': 'Avg Negative Streak Return (%)',
                }
                for i, (key, label) in enumerate(streak_map.items()):
                    if key in results:
                        with streak_cols[i % 4]:
                            st.metric(label, f"{results[key]:.2f}" if not pd.isna(results[key]) else "N/A")
                
                # Show streak frequencies
                if 'Streak_Positive_Streak_Frequencies' in results and results['Streak_Positive_Streak_Frequencies']:
                    st.subheader("📊 Streak Frequencies")
                    freq_cols = st.columns(2)
                    
                    with freq_cols[0]:
                        st.markdown("**Positive Streak Frequencies**")
                        pos_freq = results['Streak_Positive_Streak_Frequencies']
                        pos_returns = results.get('Streak_Positive_Streak_Returns', {})
                        
                        # Calcola totali per percentuali
                        total_positive_days = sum(pos_freq.values())
                        total_all_days = results['data_points']
                        
                        for length in sorted(pos_freq.keys()):
                            count = pos_freq[length]
                            avg_return = pos_returns.get(length, np.nan)
                            
                            # Calcola entrambe le percentuali
                            pct_of_positive = (count / total_positive_days) * 100 if total_positive_days > 0 else 0
                            pct_of_all = (count / total_all_days) * 100 if total_all_days > 0 else 0
                            
                            if not pd.isna(avg_return):
                                st.write(f"{length} periods: {pct_of_positive:.1f}% of positive days, {pct_of_all:.1f}% of all days, avg return: {avg_return:.2f}%")
                            else:
                                st.write(f"{length} periods: {pct_of_positive:.1f}% of positive days, {pct_of_all:.1f}% of all days")
                    
                    with freq_cols[1]:
                        st.markdown("**Negative Streak Frequencies**")
                        neg_freq = results.get('Streak_Negative_Streak_Frequencies', {})
                        neg_returns = results.get('Streak_Negative_Streak_Returns', {})
                        
                        # Calcola totali per percentuali
                        total_negative_days = sum(neg_freq.values()) if neg_freq else 0
                        total_all_days = results['data_points']
                        
                        for length in sorted(neg_freq.keys()):
                            count = neg_freq[length]
                            avg_return = neg_returns.get(length, np.nan)
                            
                            # Calcola entrambe le percentuali
                            pct_of_negative = (count / total_negative_days) * 100 if total_negative_days > 0 else 0
                            pct_of_all = (count / total_all_days) * 100 if total_all_days > 0 else 0
                            
                            if not pd.isna(avg_return):
                                st.write(f"{length} periods: {pct_of_negative:.1f}% of negative days, {pct_of_all:.1f}% of all days, avg return: {avg_return:.2f}%")
                            else:
                                st.write(f"{length} periods: {pct_of_negative:.1f}% of negative days, {pct_of_all:.1f}% of all days")
                
                st.subheader("🏆 Best/Worst Periods")
                best_worst_cols = st.columns(4)
                best_worst_map = {
                    'BestWorst_Best_Single_Period': 'Best Single Period (%)',
                    'BestWorst_Worst_Single_Period': 'Worst Single Period (%)',
                    'MeanRev_Best_Streak_Return': 'Best Streak Return (%)',
                    'MeanRev_Worst_Streak_Return': 'Worst Streak Return (%)',
                }
                for i, (key, label) in enumerate(best_worst_map.items()):
                    if key in results:
                        with best_worst_cols[i % 4]:
                            value = results[key]
                            color = "normal" if pd.isna(value) else ("inverse" if "Worst" in label else "normal")
                            st.metric(label, f"{value:.2f}" if not pd.isna(value) else "N/A", delta_color=color)
                
                st.subheader("⚡ Extreme Events")
                
                # Mostra le soglie percentuali in modo organizzato
                for threshold in [1, 3, 5]:
                    st.markdown(f"**Movimenti > {threshold}%:**")
                    extreme_cols = st.columns(4)
                    
                    # Preparazione delle metriche per questa soglia
                    threshold_metrics = {
                        f'Extreme_Days_Above_{threshold}pct': f'Days > +{threshold}%',
                        f'Extreme_Days_Below_{threshold}pct': f'Days < -{threshold}%',
                        f'Extreme_Days_Above_{threshold}pct_Percentage': f'Days > +{threshold}% (%)',
                        f'Extreme_Days_Below_{threshold}pct_Percentage': f'Days < -{threshold}% (%)',
                    }
                    
                    for i, (key, label) in enumerate(threshold_metrics.items()):
                        if key in results:
                            with extreme_cols[i % 4]:
                                value = results[key]
                                if 'Percentage' in key:
                                    display_val = f"{value:.2f}" if not pd.isna(value) else "N/A"
                                else:
                                    display_val = f"{int(value)}" if not pd.isna(value) else "N/A"
                                st.metric(label, display_val)
                
                # Deviazione standard come riferimento
                if 'Extreme_Return_Std_Dev' in results:
                    st.markdown("**Riferimento Statistico:**")
                    ref_col = st.columns(1)[0]
                    with ref_col:
                        st.metric("Std Deviation (%)", f"{results['Extreme_Return_Std_Dev']:.2f}" if not pd.isna(results['Extreme_Return_Std_Dev']) else "N/A")
                
                st.subheader("🔄 Mean Reversion Analysis")
                mean_rev_cols = st.columns(3)
                mean_rev_map = {
                    'MeanRev_Avg_Periods_To_MA9': 'Avg Periods to MA9',
                    'MeanRev_Avg_Periods_To_Recovery': 'Avg Periods to Recovery',
                    'MeanRev_Avg_Overshoot_MA9': 'Avg Overshoot MA9 (%)',
                }
                for i, (key, label) in enumerate(mean_rev_map.items()):
                    if key in results:
                        with mean_rev_cols[i % 3]:
                            st.metric(label, f"{results[key]:.2f}" if not pd.isna(results[key]) else "N/A")
                
                st.subheader("📈 Volatility & Volume Analysis")
                vol_cols = st.columns(3)
                vol_map = {
                    'Volatility_ATR_Mean': 'Average ATR',
                    'Volatility_Range_Mean': 'Average Range',
                    'Volatility_Volatility_Clustering': 'Volatility Clustering (%)',
                }
                for i, (key, label) in enumerate(vol_map.items()):
                    if key in results:
                        with vol_cols[i % 3]:
                            st.metric(label, f"{results[key]:.2f}" if not pd.isna(results[key]) else "N/A")
                
                # Nuova sezione per Range Metrics
                st.subheader("📏 Range Analysis")
                
                # Range base metrics
                range_base_cols = st.columns(3)
                range_base_map = {
                    'Range_Range_Mean_Absolute': 'Average Range (Abs)',
                    'Range_Range_Mean_Percentage': 'Average Range (%)', 
                    'Range_Range_Volatility': 'Range Volatility',
                }
                for i, (key, label) in enumerate(range_base_map.items()):
                    if key in results:
                        with range_base_cols[i % 3]:
                            value = results[key]
                            if key == 'Range_Range_Volatility':
                                # Aggiungi indicatore di livello per Range Volatility
                                level = results.get('Range_Range_Volatility_Level', 'N/A')
                                level_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(level, "⚪")
                                st.metric(f"{label} {level_emoji}", f"{value:.2f}" if not pd.isna(value) else "N/A", 
                                        delta=f"{level} volatility" if level != "N/A" else None)
                            else:
                                st.metric(label, f"{value:.2f}" if not pd.isna(value) else "N/A")
                
                # Range timeframe-specific metrics
                if timeframe != "Daily":
                    st.markdown(f"**{timeframe} Range Metrics:**")
                    range_timeframe_cols = st.columns(3)
                    
                    timeframe_metrics = {}
                    if timeframe == "Weekly":
                        timeframe_metrics = {
                            'Range_Weekly_Range_Mean': 'Average Weekly Range',
                            'Range_Daily_Range_Per_Week_Day': 'Daily Range per Week Day',
                            'Range_Cumulative_vs_Period_Range_Ratio': 'Cumulative vs Period Ratio',
                        }
                    elif timeframe == "Monthly":
                        timeframe_metrics = {
                            'Range_Monthly_Range_Mean': 'Average Monthly Range',
                            'Range_Daily_Range_Per_Month_Day': 'Daily Range per Month Day',
                            'Range_Cumulative_vs_Period_Range_Ratio': 'Cumulative vs Period Ratio',
                        }
                    elif timeframe == "Yearly":
                        timeframe_metrics = {
                            'Range_Yearly_Range_Mean': 'Average Yearly Range',
                            'Range_Daily_Range_Per_Year_Day': 'Daily Range per Year Day',
                            'Range_Cumulative_vs_Period_Range_Ratio': 'Cumulative vs Period Ratio',
                        }
                    
                    for i, (key, label) in enumerate(timeframe_metrics.items()):
                        if key in results:
                            with range_timeframe_cols[i % 3]:
                                st.metric(label, f"{results[key]:.2f}" if not pd.isna(results[key]) else "N/A")
                
                # Range position analysis
                st.markdown("**Range Position Analysis:**")
                range_pos_cols = st.columns(4)
                range_pos_map = {
                    'Range_Avg_Close_Range_Position': 'Avg Close Position in Range (%)',
                    'Range_Range_Close_Near_High_Pct': 'Close Near High (%)',
                    'Range_Range_Close_Near_Low_Pct': 'Close Near Low (%)',
                    'Range_Range_Position_Bias': 'Position Bias',
                }
                for i, (key, label) in enumerate(range_pos_map.items()):
                    if key in results:
                        with range_pos_cols[i % 4]:
                            value = results[key]
                            if key == 'Range_Range_Position_Bias':
                                # Aggiungi emoji per il bias
                                bias_emoji = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(str(value), "⚪")
                                st.metric(f"{label} {bias_emoji}", str(value))
                            else:
                                # Aggiungi context per valori 0
                                display_val = f"{value:.1f}" if not pd.isna(value) else "N/A"
                                help_text = None
                                
                                if "Close Near" in label and value == 0.0:
                                    help_text = "Nessun giorno con close in questa zona del range nel periodo selezionato"
                                elif "Close Near High" in label:
                                    help_text = "Percentuale di giorni con close >80% del range giornaliero"
                                elif "Close Near Low" in label:
                                    help_text = "Percentuale di giorni con close <20% del range giornaliero"
                                
                                st.metric(label, display_val, help=help_text)
                
                # Range efficiency
                if 'Range_Range_Efficiency' in results:
                    st.markdown("**Range Efficiency:**")
                    eff_col = st.columns(1)[0]
                    with eff_col:
                        st.metric("Range Efficiency (%)", f"{results['Range_Range_Efficiency']:.1f}" if not pd.isna(results['Range_Range_Efficiency']) else "N/A", 
                                help="Quanto del range giornaliero viene utilizzato dai movimenti di prezzo (Open->Close)")
                
                st.subheader("📊 Volume Analysis")
                volume_cols = st.columns(4)
                volume_map = {
                    'Volume_Volume_Mean': 'Average Volume',
                    'Volume_Days_Above_Avg_Volume_Pct': 'Days Above Avg Vol (%)',
                    'Volume_Positive_Days_Avg_Volume': 'Positive Days Avg Volume',
                    'Volume_Negative_Days_Avg_Volume': 'Negative Days Avg Volume',
                }
                for i, (key, label) in enumerate(volume_map.items()):
                    if key in results:
                        with volume_cols[i % 4]:
                            value = results[key]
                            if 'Volume_Mean' in key or 'Volume' in label:
                                display_val = f"{value:,.0f}" if not pd.isna(value) else "N/A"
                            else:
                                display_val = f"{value:.2f}" if not pd.isna(value) else "N/A"
                            st.metric(label, display_val)
                
                st.subheader("🔗 Autocorrelation Analysis")
                autocorr_cols = st.columns(3)
                autocorr_keys = [k for k in results.keys() if k.startswith('Autocorr_')]
                for i, key in enumerate(autocorr_keys):
                    with autocorr_cols[i % 3]:
                        lag = key.split('_')[1]
                        st.metric(f"Autocorrelation {lag}", f"{results[key]:.3f}" if not pd.isna(results[key]) else "N/A")
                
                st.subheader("🎭 Price Action Patterns")
                
                # Pattern recognition
                st.markdown("**Candlestick & Chart Patterns:**")
                pattern_cols = st.columns(3)
                pattern_map = {
                    'PriceAction_PriceAction_Doji_Percentage': 'Doji Days (%)',
                    'PriceAction_PriceAction_Hammer_Percentage': 'Hammer Days (%)',
                    'PriceAction_PriceAction_ShootingStar_Percentage': 'Shooting Star Days (%)',
                }
                for i, (key, label) in enumerate(pattern_map.items()):
                    if key in results:
                        with pattern_cols[i % 3]:
                            st.metric(label, f"{results[key]:.2f}" if not pd.isna(results[key]) else "N/A")
                
                # Gap analysis
                st.markdown("**Gap Analysis:**")
                gap_cols = st.columns(4)
                gap_map = {
                    'PriceAction_PriceAction_Gap_Up_Days': 'Gap Up Days',
                    'PriceAction_PriceAction_Gap_Down_Days': 'Gap Down Days',
                    'PriceAction_PriceAction_Gap_Up_Avg_Size': 'Avg Gap Up Size (%)',
                    'PriceAction_PriceAction_Gap_Down_Avg_Size': 'Avg Gap Down Size (%)',
                }
                for i, (key, label) in enumerate(gap_map.items()):
                    if key in results:
                        with gap_cols[i % 4]:
                            value = results[key]
                            if 'Days' in label:
                                display_val = f"{int(value)}" if not pd.isna(value) else "N/A"
                            else:
                                display_val = f"{value:.2f}" if not pd.isna(value) else "N/A"
                            st.metric(label, display_val)
                
                # Gap fill rates
                gap_fill_cols = st.columns(2)
                gap_fill_map = {
                    'PriceAction_PriceAction_Gap_Up_Fill_Rate': 'Gap Up Fill Rate (%)',
                    'PriceAction_PriceAction_Gap_Down_Fill_Rate': 'Gap Down Fill Rate (%)',
                }
                for i, (key, label) in enumerate(gap_fill_map.items()):
                    if key in results:
                        with gap_fill_cols[i]:
                            st.metric(label, f"{results[key]:.1f}" if not pd.isna(results[key]) else "N/A")
                
                # Inside/Outside days
                st.markdown("**Range Relationship Patterns:**")
                range_pattern_cols = st.columns(4)
                range_pattern_map = {
                    'PriceAction_PriceAction_Inside_Days_Percentage': 'Inside Days (%)',
                    'PriceAction_PriceAction_Outside_Days_Percentage': 'Outside Days (%)',
                    'PriceAction_PriceAction_Strong_Close_Percentage': 'Strong Closes (%)',
                    'PriceAction_PriceAction_Weak_Close_Percentage': 'Weak Closes (%)',
                }
                for i, (key, label) in enumerate(range_pattern_map.items()):
                    if key in results:
                        with range_pattern_cols[i % 4]:
                            st.metric(label, f"{results[key]:.2f}" if not pd.isna(results[key]) else "N/A")
                
                # Efficacia dei pattern
                st.markdown("**Pattern Effectiveness:**")
                effectiveness_cols = st.columns(2)
                effectiveness_map = {
                    'PriceAction_PriceAction_Strong_Close_Next_Day_Success': 'Strong Close → Next Day Success (%)',
                    'PriceAction_PriceAction_Weak_Close_Next_Day_Success': 'Weak Close → Next Day Success (%)',
                }
                for i, (key, label) in enumerate(effectiveness_map.items()):
                    if key in results:
                        with effectiveness_cols[i]:
                            value = results[key]
                            color = "normal"
                            if not pd.isna(value):
                                if value > 60:
                                    color = "normal"
                                elif value < 40:
                                    color = "inverse"
                            st.metric(label, f"{value:.1f}" if not pd.isna(value) else "N/A", delta_color=color)
                
                # ===== WEEKDAY BEHAVIOR ANALYSIS =====
                if timeframe == "Daily":
                    weekday_keys = [k for k in results.keys() if k.startswith('Weekday_')]
                    if weekday_keys:
                        st.subheader("📅 Weekday Behavior Analysis")
                        
                        # 1. Analisi Settimanale Base
                        st.markdown("**📊 Analisi Performance Settimanale**")
                        weekly_cols = st.columns(4)
                        weekly_base_map = {
                            'Weekday_Prob_Week_Positive_Given_Monday_Positive': 'Sett+ se Lun+ (%)',
                            'Weekday_Prob_Week_Negative_Given_Monday_Positive': 'Sett- se Lun+ (%)',
                            'Weekday_Prob_Week_Positive_Given_Monday_Negative': 'Sett+ se Lun- (%)',
                            'Weekday_Prob_Week_Negative_Given_Monday_Negative': 'Sett- se Lun- (%)',
                        }
                        
                        for i, (key, label) in enumerate(weekly_base_map.items()):
                            if key in results:
                                with weekly_cols[i % 4]:
                                    value = results[key]
                                    delta_color = "normal"
                                    if not pd.isna(value):
                                        if value > 60:
                                            delta_color = "normal"
                                        elif value < 40:
                                            delta_color = "inverse"
                                    st.metric(label, f"{value:.1f}" if not pd.isna(value) else "N/A", delta_color=delta_color)
                        
                        # 2. Statistiche Base
                        st.markdown("**📈 Statistiche Generali**")
                        stats_cols = st.columns(4)
                        stats_map = {
                            'Weekday_Total_Weeks_Analyzed': 'Settimane Analizzate',
                            'Weekday_Overall_Weekly_Positive_Rate': 'Rate Sett+ Generale (%)',
                            'Weekday_Overall_Monday_Positive_Rate': 'Rate Lun+ Generale (%)',
                            'Weekday_Monday_Weekly_Return_Correlation': 'Correlazione Lun-Sett',
                        }
                        
                        for i, (key, label) in enumerate(stats_map.items()):
                            if key in results:
                                with stats_cols[i % 4]:
                                    value = results[key]
                                    if 'Correlation' in label:
                                        display_val = f"{value:.3f}" if not pd.isna(value) else "N/A"
                                    elif 'Rate' in label:
                                        display_val = f"{value:.1f}" if not pd.isna(value) else "N/A"
                                    else:
                                        display_val = f"{int(value)}" if not pd.isna(value) else "N/A"
                                    st.metric(label, display_val)
                        
                        # 3. Analisi Lunedì → Altri Giorni
                        st.markdown("**🔄 Lunedì → Altri Giorni della Settimana**")
                        
                        days = ['Tuesday', 'Wednesday', 'Thursday', 'Friday']
                        day_names_it = ['Martedì', 'Mercoledì', 'Giovedì', 'Venerdì']
                        
                        for i, (day, day_it) in enumerate(zip(days, day_names_it)):
                            st.markdown(f"**{day_it}:**")
                            day_cols = st.columns(4)
                            
                            day_prob_map = {
                                f'Weekday_Prob_{day}_Positive_Given_Monday_Positive': f'{day_it}+ se Lun+ (%)',
                                f'Weekday_Prob_{day}_Negative_Given_Monday_Positive': f'{day_it}- se Lun+ (%)',
                                f'Weekday_Prob_{day}_Positive_Given_Monday_Negative': f'{day_it}+ se Lun- (%)',
                                f'Weekday_Prob_{day}_Negative_Given_Monday_Negative': f'{day_it}- se Lun- (%)',
                            }
                            
                            for j, (key, label) in enumerate(day_prob_map.items()):
                                if key in results:
                                    with day_cols[j % 4]:
                                        value = results[key]
                                        delta_color = "normal"
                                        if not pd.isna(value):
                                            if value > 55:
                                                delta_color = "normal"
                                            elif value < 45:
                                                delta_color = "inverse"
                                        st.metric(label, f"{value:.1f}" if not pd.isna(value) else "N/A", delta_color=delta_color)
                            
                            # Coppie analizzate
                            pairs_key = f'Weekday_{day}_Monday_Pairs_Analyzed'
                            if pairs_key in results:
                                st.caption(f"Coppie Lunedì-{day_it} analizzate: {int(results[pairs_key])}")
                        
                        # 4. Analisi Venerdì Precedente → Lunedì
                        st.markdown("**🔄 Venerdì Precedente → Lunedì Successivo**")
                        friday_monday_cols = st.columns(4)
                        friday_monday_map = {
                            'Weekday_Prob_Monday_Positive_Given_Friday_Previous_Positive': 'Lun+ se Ven prec+ (%)',
                            'Weekday_Prob_Monday_Negative_Given_Friday_Previous_Positive': 'Lun- se Ven prec+ (%)',
                            'Weekday_Prob_Monday_Positive_Given_Friday_Previous_Negative': 'Lun+ se Ven prec- (%)',
                            'Weekday_Prob_Monday_Negative_Given_Friday_Previous_Negative': 'Lun- se Ven prec- (%)',
                        }
                        
                        for i, (key, label) in enumerate(friday_monday_map.items()):
                            if key in results:
                                with friday_monday_cols[i % 4]:
                                    value = results[key]
                                    delta_color = "normal"
                                    if not pd.isna(value):
                                        if value > 55:
                                            delta_color = "normal"
                                        elif value < 45:
                                            delta_color = "inverse"
                                    st.metric(label, f"{value:.1f}" if not pd.isna(value) else "N/A", delta_color=delta_color)
                        
                        # Statistiche Venerdì-Lunedì
                        friday_stats_cols = st.columns(3)
                        friday_stats_map = {
                            'Weekday_Friday_Monday_Consecutive_Pairs_Analyzed': 'Coppie Ven-Lun Analizzate',
                            'Weekday_Friday_Previous_Positive_Count': 'Venerdì Precedenti Positivi',
                            'Weekday_Friday_Previous_Negative_Count': 'Venerdì Precedenti Negativi',
                        }
                        
                        for i, (key, label) in enumerate(friday_stats_map.items()):
                            if key in results:
                                with friday_stats_cols[i % 3]:
                                    value = results[key]
                                    st.metric(label, f"{int(value)}" if not pd.isna(value) else "N/A")
                        
                        # 5. Insights Summary
                        st.markdown("**💡 Key Insights**")
                        
                        # Calcola insights automatici
                        insights = []
                        
                        # Insight 1: Settimana vs Lunedì
                        week_pos_mon_pos = results.get('Weekday_Prob_Week_Positive_Given_Monday_Positive', 0)
                        week_pos_mon_neg = results.get('Weekday_Prob_Week_Positive_Given_Monday_Negative', 0)
                        if not pd.isna(week_pos_mon_pos) and not pd.isna(week_pos_mon_neg):
                            if week_pos_mon_pos > 60:
                                insights.append(f"📈 **Forte correlazione positiva**: Se il Lunedì è positivo, la settimana è positiva nel {week_pos_mon_pos:.1f}% dei casi")
                            if week_pos_mon_neg < 40:
                                insights.append(f"📉 **Effetto negativo persistente**: Se il Lunedì è negativo, la settimana è positiva solo nel {week_pos_mon_neg:.1f}% dei casi")
                        
                        # Insight 2: Mean reversion Venerdì-Lunedì
                        mon_pos_fri_pos = results.get('Weekday_Prob_Monday_Positive_Given_Friday_Previous_Positive', 0)
                        mon_pos_fri_neg = results.get('Weekday_Prob_Monday_Positive_Given_Friday_Previous_Negative', 0)
                        if not pd.isna(mon_pos_fri_pos) and not pd.isna(mon_pos_fri_neg):
                            if mon_pos_fri_neg > mon_pos_fri_pos:
                                insights.append(f"🔄 **Mean Reversion Weekend**: Dopo Venerdì negativo, Lunedì è positivo nel {mon_pos_fri_neg:.1f}% vs {mon_pos_fri_pos:.1f}% dopo Venerdì positivo")
                        
                        # Insight 3: Correlazione generale
                        correlation = results.get('Weekday_Monday_Weekly_Return_Correlation', 0)
                        if not pd.isna(correlation):
                            if correlation > 0.3:
                                insights.append(f"🔗 **Correlazione moderata-forte**: Correlazione Lunedì-Settimana di {correlation:.3f}")
                            elif correlation < 0.1:
                                insights.append(f"🎲 **Bassa predittività**: Correlazione Lunedì-Settimana molto bassa ({correlation:.3f})")
                        
                        for insight in insights:
                            st.markdown(insight)
                        
                        if not insights:
                            st.info("💭 Analizza i dati per identificare pattern interessanti nei comportamenti settimanali")
                
                # Export functionality
                st.subheader("💾 Export Data")
                
                # Create export dataframe
                filter_desc = []
                for k, v in filters.items():
                    if v not in [None, [], "All"]:
                        if isinstance(v, list):
                            filter_desc.append(f"{k}: {', '.join(map(str, v))}")
                        else:
                            filter_desc.append(f"{k}: {v}")
                
                filter_string = "; ".join(filter_desc) if filter_desc else "No filters"
                
                row_data = {
                    'Timeframe': timeframe,
                    'Filters': filter_string,
                    'Data_Points': results['data_points'],
                    'Date_Range': results['date_range']
                }
                
                # Add all metrics to export
                for key, value in results.items():
                    if key not in ['timeframe', 'filters_applied', 'data_points', 'date_range']:
                        if isinstance(value, dict):
                            continue  # Skip frequency dictionaries for now
                        row_data[key] = value
                
                export_df = pd.DataFrame([row_data])
                
                # Export buttons
                export_cols = st.columns(2)
                with export_cols[0]:
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"oil_analysis_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with export_cols[1]:
                    json_data = export_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON", 
                        data=json_data,
                        file_name=f"oil_analysis_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
        except Exception as e:
            st.error(f"Errore nel processing dei dati: {str(e)}")
            st.info("Verifica che il dataset contenga le colonne richieste")
            
            # Debug info
            if df is not None:
                try:
                    st.write("**Colonne trovate nel dataset:**")
                    st.write(list(df.columns))
                    st.write("**Prime 3 righe:**")
                    st.write(df.head(3))
                except Exception:
                    pass
    
    else:
        if data_source == "Carica Nuovo File CSV":
            st.info("👆 Carica un file CSV per iniziare l'analisi")
        else:
            st.warning("⚠️ Dataset pre-incluso non disponibile. Prova a caricare un file CSV.")
        
        # Mostra esempio di struttura dati richiesta
        st.subheader("📋 Struttura dati richiesta")
        example_columns = [
            "Date", "Open", "High", "Low", "Close", "Volume", 
            "Quarter", "Day_of_Week", "Month", "Week_of_Month",
            "Change_Pct", "Daily_Range", "ATR_14", "MA7"
        ]
        st.write("Il CSV deve contenere almeno queste colonne:")
        st.code(", ".join(example_columns))
        
        st.markdown("""
        **Colonne principali richieste:**
        - `Date`: Date in formato YYYY-MM-DD
        - `Open, High, Low, Close`: Prezzi OHLC
        - `Volume`: Volume giornaliero
        - `Change_Pct`: Return percentuale giornaliero
        - `Quarter`: Quarter (es: Q1, Q2, Q3, Q4)
        - `Day_of_Week`: Giorno della settimana
        - `Month`: Numero del mese (1-12)
        - `Week_of_Month`: Settimana del mese (1-5)
        - `Daily_Range`: Range giornaliero High-Low
        - `ATR_14`: Average True Range 14 periodi
        """)

# Esecuzione dell'app
if __name__ == "__main__":
    create_streamlit_app()
