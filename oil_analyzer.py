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
                # Il tuo dataset ha Quarter come string (Q1, Q2, Q3, Q4)
                filtered_df = filtered_df[filtered_df['Quarter'].isin(filter_value)]
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
                    'Daily_Range': group['Daily_Range'].sum(),
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
                    'Daily_Range': group['Daily_Range'].sum(),
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
                    'Daily_Range': group['Daily_Range'].sum(),
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
        """Calcola extreme events (2σ)"""
        if len(df) == 0:
            return {}
        
        returns = df['Return']
        std_dev = returns.std()
        mean_return = returns.mean()
        
        threshold_up = mean_return + 2 * std_dev
        threshold_down = mean_return - 2 * std_dev
        
        days_above_2sigma = (returns > threshold_up).sum()
        days_below_2sigma = (returns < threshold_down).sum()
        
        return {
            'Days_Above_2Sigma': days_above_2sigma,
            'Days_Below_2Sigma': days_below_2sigma,
            'Days_Above_2Sigma_Pct': (days_above_2sigma / len(df)) * 100,
            'Days_Below_2Sigma_Pct': (days_below_2sigma / len(df)) * 100,
            'Return_Std_Dev': std_dev,
        }
    
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
                available_quarters = sorted(df['Quarter'].unique())
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
                extreme_cols = st.columns(5)
                extreme_map = {
                    'Extreme_Days_Above_2Sigma': 'Days > 2σ',
                    'Extreme_Days_Below_2Sigma': 'Days < -2σ',
                    'Extreme_Days_Above_2Sigma_Pct': 'Days > 2σ (%)',
                    'Extreme_Days_Below_2Sigma_Pct': 'Days < -2σ (%)',
                    'Extreme_Return_Std_Dev': 'Std Deviation (%)',
                }
                for i, (key, label) in enumerate(extreme_map.items()):
                    if key in results:
                        with extreme_cols[i % 5]:
                            value = results[key]
                            if 'Pct' in key or 'Std_Dev' in key:
                                display_val = f"{value:.2f}" if not pd.isna(value) else "N/A"
                            else:
                                display_val = f"{int(value)}" if not pd.isna(value) else "N/A"
                            st.metric(label, display_val)
                
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

if __name__ == "__main__":
    create_streamlit_app()
    
