#!/usr/bin/env python3
"""
Program 1: Core Oil Analyzer
Analizza TUTTO il dataset e genera automaticamente la struttura gerarchica completa:
- total_program1.json (tutto il dataset)
- YYYY/YYYY_program1.json (ogni anno)
- YYYY/QX/YYYY-QX_program1.json (ogni trimestre)
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

class CoreOilAnalyzer:
    def __init__(self, data_file):
        """Inizializza l'analyzer con il file dati"""
        print(f"📊 Loading data from {data_file}...")
        self.df = pd.read_csv(data_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        print(f"✅ Loaded {len(self.df)} records from {self.df['Date'].min().strftime('%Y-%m-%d')} to {self.df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Rileva automaticamente tutti i timeframe disponibili
        self.timeframes = self._detect_timeframes()
        
    def _detect_timeframes(self):
        """Rileva automaticamente tutti i timeframe disponibili nei dati"""
        timeframes = {}
        
        # Total dataset
        timeframes['total'] = len(self.df)
        
        # Per ogni anno
        years = sorted(self.df['Date'].dt.year.unique())
        timeframes['years'] = {}
        
        for year in years:
            year_data = self.df[self.df['Date'].dt.year == year]
            timeframes['years'][year] = {
                'count': len(year_data),
                'quarters': {}
            }
            
            # Per ogni trimestre dell'anno
            quarters = sorted(year_data['Quarter'].unique())
            for quarter in quarters:
                if f"{year}-Q" in quarter:  # Es: "2020-Q1"
                    quarter_num = quarter.split('-Q')[1]
                    quarter_data = year_data[year_data['Quarter'] == quarter]
                    timeframes['years'][year]['quarters'][quarter_num] = {
                        'count': len(quarter_data),
                        'period_code': quarter
                    }
        
        print(f"🎯 Detected timeframes:")
        print(f"   - Total dataset: {timeframes['total']} records")
        print(f"   - Years: {len(timeframes['years'])} ({min(years)}-{max(years)})")
        total_quarters = sum(len(year_info['quarters']) for year_info in timeframes['years'].values())
        print(f"   - Quarters: {total_quarters} total")
        
        return timeframes
    
    def get_data_quality(self, df):
        """Calcola la qualità dei dati"""
        total_days = len(df)
        
        # Controlla dati mancanti per colonne chiave
        key_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change_Pct']
        missing_data = df[key_columns].isnull().sum().sum()
        complete_percentage = ((total_days * len(key_columns) - missing_data) / 
                             (total_days * len(key_columns)) * 100)
        
        # Date mancanti (assumendo trading days)
        if total_days > 0:
            date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
            expected_trading_days = len([d for d in date_range if d.weekday() < 5])  # Mon-Fri
            missing_dates = max(0, expected_trading_days - total_days)
        else:
            missing_dates = 0
        
        # Outliers detection (IQR method per Change_Pct)
        if len(df) > 0:
            Q1 = df['Change_Pct'].quantile(0.25)
            Q3 = df['Change_Pct'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df['Change_Pct'] < Q1 - 3*IQR) | (df['Change_Pct'] > Q3 + 3*IQR)]
            outliers_count = len(outliers)
        else:
            outliers_count = 0
        
        return {
            "total_days": total_days,
            "complete_data_percentage": round(complete_percentage, 2),
            "missing_dates_estimate": missing_dates,
            "outliers_detected": outliers_count
        }
    
    def analyze_basic_performance(self, df):
        """Analizza performance di base"""
        if len(df) == 0:
            return self._empty_basic_performance()
            
        result = {}
        
        # Conteggi giorni positivi/negativi
        result["days_positive"] = int(df[df['Is_Positive'] == 1].shape[0])
        result["days_negative"] = int(df[df['Is_Positive'] == 0].shape[0])
        
        # Streak analysis
        positive_streaks = []
        negative_streaks = []
        current_streak = 0
        current_type = None
        
        for _, row in df.iterrows():
            if row['Is_Positive'] == 1:
                if current_type == 'positive':
                    current_streak += 1
                else:
                    if current_type == 'negative' and current_streak > 0:
                        negative_streaks.append(current_streak)
                    current_streak = 1
                    current_type = 'positive'
            else:
                if current_type == 'negative':
                    current_streak += 1
                else:
                    if current_type == 'positive' and current_streak > 0:
                        positive_streaks.append(current_streak)
                    current_streak = 1
                    current_type = 'negative'
        
        # Aggiungi l'ultimo streak
        if current_type == 'positive' and current_streak > 0:
            positive_streaks.append(current_streak)
        elif current_type == 'negative' and current_streak > 0:
            negative_streaks.append(current_streak)
        
        # Top 5 streaks
        positive_streaks.sort(reverse=True)
        negative_streaks.sort(reverse=True)
        
        result["max_positive_streak_top5"] = positive_streaks[:5] if positive_streaks else [0]
        result["max_negative_streak_top5"] = negative_streaks[:5] if negative_streaks else [0]
        result["max_positive_streak"] = max(positive_streaks) if positive_streaks else 0
        result["max_negative_streak"] = max(negative_streaks) if negative_streaks else 0
        
        # Rally/Decline analysis
        if len(df) > 0:
            changes_sorted = df.sort_values('Change_Pct', ascending=False)
            
            result["max_rally_pct_top5"] = changes_sorted['Change_Pct'].head(5).round(4).tolist()
            result["max_decline_pct_top5"] = changes_sorted['Change_Pct'].tail(5).round(4).tolist()
            result["max_rally_pct"] = round(float(df['Change_Pct'].max()), 4)
            result["max_decline_pct"] = round(float(df['Change_Pct'].min()), 4)
            
            # Medie
            result["avg_abs_change"] = round(float(df['Change_Pct'].abs().mean()), 4)
            
            positive_changes = df[df['Change_Pct'] > 0]['Change_Pct']
            negative_changes = df[df['Change_Pct'] < 0]['Change_Pct']
            
            result["avg_positive_change"] = round(float(positive_changes.mean()), 4) if len(positive_changes) > 0 else 0.0
            result["avg_negative_change"] = round(float(negative_changes.mean()), 4) if len(negative_changes) > 0 else 0.0
        else:
            result.update({
                "max_rally_pct_top5": [0],
                "max_decline_pct_top5": [0],
                "max_rally_pct": 0.0,
                "max_decline_pct": 0.0,
                "avg_abs_change": 0.0,
                "avg_positive_change": 0.0,
                "avg_negative_change": 0.0
            })
        
        return result
    
    def _empty_basic_performance(self):
        """Ritorna struttura vuota per basic_performance"""
        return {
            "days_positive": 0,
            "days_negative": 0,
            "max_positive_streak_top5": [0],
            "max_negative_streak_top5": [0],
            "max_positive_streak": 0,
            "max_negative_streak": 0,
            "max_rally_pct_top5": [0],
            "max_decline_pct_top5": [0],
            "max_rally_pct": 0.0,
            "max_decline_pct": 0.0,
            "avg_abs_change": 0.0,
            "avg_positive_change": 0.0,
            "avg_negative_change": 0.0
        }
    
    def get_top5_with_details(self, df, column, ascending=False):
        """Ottiene top 5 valori con dettagli (data, valore)"""
        if len(df) == 0:
            return []
            
        sorted_df = df.sort_values(column, ascending=ascending)
        top5 = []
        
        for i in range(min(5, len(sorted_df))):
            row = sorted_df.iloc[i]
            top5.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "value": round(float(row[column]), 4)
            })
        
        return top5
    
    def get_extreme_with_details(self, df, column, is_max=True):
        """Ottiene valore estremo con dettagli"""
        if len(df) == 0:
            return {"date": "N/A", "value": 0.0}
            
        if is_max:
            extreme_row = df.loc[df[column].idxmax()]
        else:
            extreme_row = df.loc[df[column].idxmin()]
            
        return {
            "date": extreme_row['Date'].strftime('%Y-%m-%d'),
            "value": round(float(extreme_row[column]), 4)
        }
    
    def analyze_volume_range_extremes(self, df):
        """Analizza estremi di volume e range"""
        if len(df) == 0:
            return self._empty_volume_range_extremes()
            
        result = {}
        
        # Volume analysis
        result["volume_top5_max"] = self.get_top5_with_details(df, 'Volume', ascending=False)
        result["volume_top5_min"] = self.get_top5_with_details(df, 'Volume', ascending=True)
        result["volume_max"] = self.get_extreme_with_details(df, 'Volume', is_max=True)
        result["volume_min"] = self.get_extreme_with_details(df, 'Volume', is_max=False)
        result["volume_avg"] = round(float(df['Volume'].mean()), 2)
        
        # Range analysis
        result["range_top5_max"] = self.get_top5_with_details(df, 'Daily_Range', ascending=False)
        result["range_top5_min"] = self.get_top5_with_details(df, 'Daily_Range', ascending=True)
        result["range_max"] = self.get_extreme_with_details(df, 'Daily_Range', is_max=True)
        result["range_min"] = self.get_extreme_with_details(df, 'Daily_Range', is_max=False)
        
        # Change analysis
        result["change_top5_max"] = self.get_top5_with_details(df, 'Change_Pct', ascending=False)
        result["change_top5_min"] = self.get_top5_with_details(df, 'Change_Pct', ascending=True)
        result["change_max"] = self.get_extreme_with_details(df, 'Change_Pct', is_max=True)
        result["change_min"] = self.get_extreme_with_details(df, 'Change_Pct', is_max=False)
        
        # Upper Shadow analysis
        result["upper_shadow_top5_max"] = self.get_top5_with_details(df, 'Upper_Shadow', ascending=False)
        result["upper_shadow_top5_min"] = self.get_top5_with_details(df, 'Upper_Shadow', ascending=True)
        result["upper_shadow_max"] = self.get_extreme_with_details(df, 'Upper_Shadow', is_max=True)
        result["upper_shadow_min"] = self.get_extreme_with_details(df, 'Upper_Shadow', is_max=False)
        
        # Lower Shadow analysis
        result["lower_shadow_top5_max"] = self.get_top5_with_details(df, 'Lower_Shadow', ascending=False)
        result["lower_shadow_top5_min"] = self.get_top5_with_details(df, 'Lower_Shadow', ascending=True)
        result["lower_shadow_max"] = self.get_extreme_with_details(df, 'Lower_Shadow', is_max=True)
        result["lower_shadow_min"] = self.get_extreme_with_details(df, 'Lower_Shadow', is_max=False)
        
        return result
    
    def _empty_volume_range_extremes(self):
        """Ritorna struttura vuota per volume_range_extremes"""
        empty_detail = {"date": "N/A", "value": 0.0}
        return {
            "volume_top5_max": [],
            "volume_top5_min": [],
            "volume_max": empty_detail,
            "volume_min": empty_detail,
            "volume_avg": 0.0,
            "range_top5_max": [],
            "range_top5_min": [],
            "range_max": empty_detail,
            "range_min": empty_detail,
            "change_top5_max": [],
            "change_top5_min": [],
            "change_max": empty_detail,
            "change_min": empty_detail,
            "upper_shadow_top5_max": [],
            "upper_shadow_top5_min": [],
            "upper_shadow_max": empty_detail,
            "upper_shadow_min": empty_detail,
            "lower_shadow_top5_max": [],
            "lower_shadow_top5_min": [],
            "lower_shadow_max": empty_detail,
            "lower_shadow_min": empty_detail
        }
    
    def analyze_ma_analysis(self, df):
        """Analizza Moving Averages"""
        if len(df) == 0:
            return self._empty_ma_analysis()
            
        result = {}
        
        # Distance analysis per ogni MA
        ma_columns = ['MA7_Distance', 'MA20_Distance', 'MA50_Distance', 'MA200_Distance']
        
        for ma_col in ma_columns:
            ma_name = ma_col.lower().replace('_distance', '')
            result[f"{ma_name}_distance_top5_max"] = self.get_top5_with_details(df, ma_col, ascending=False)
            result[f"{ma_name}_distance_max"] = self.get_extreme_with_details(df, ma_col, is_max=True)
        
        # MA Alignment (tutte le MA in ordine crescente o decrescente)
        ma_values = ['MA7', 'MA20', 'MA50', 'MA200']
        aligned_days = 0
        
        for _, row in df.iterrows():
            ma_vals = [row[ma] for ma in ma_values if not pd.isna(row[ma])]
            if len(ma_vals) == 4:
                # Controlla se sono in ordine crescente o decrescente
                is_ascending = all(ma_vals[i] <= ma_vals[i+1] for i in range(len(ma_vals)-1))
                is_descending = all(ma_vals[i] >= ma_vals[i+1] for i in range(len(ma_vals)-1))
                if is_ascending or is_descending:
                    aligned_days += 1
        
        result["ma_aligned_days"] = aligned_days
        
        # MA7-MA200 Divergence
        df_copy = df.copy()
        df_copy['ma7_ma200_divergence'] = abs(df_copy['MA7'] - df_copy['MA200'])
        
        result["ma7_ma200_divergence_top5"] = self.get_top5_with_details(df_copy, 'ma7_ma200_divergence', ascending=False)
        result["ma7_ma200_divergence_max"] = self.get_extreme_with_details(df_copy, 'ma7_ma200_divergence', is_max=True)
        
        return result
    
    def _empty_ma_analysis(self):
        """Ritorna struttura vuota per ma_analysis"""
        empty_detail = {"date": "N/A", "value": 0.0}
        return {
            "ma7_distance_top5_max": [],
            "ma7_distance_max": empty_detail,
            "ma20_distance_top5_max": [],
            "ma20_distance_max": empty_detail,
            "ma50_distance_top5_max": [],
            "ma50_distance_max": empty_detail,
            "ma200_distance_top5_max": [],
            "ma200_distance_max": empty_detail,
            "ma_aligned_days": 0,
            "ma7_ma200_divergence_top5": [],
            "ma7_ma200_divergence_max": empty_detail
        }
    
    def analyze_seasonal_patterns(self, df):
        """Analizza pattern stagionali"""
        if len(df) == 0:
            return self._empty_seasonal_patterns()
            
        result = {}
        
        # Day of Week analysis
        if 'Day_of_Week' in df.columns and len(df) > 0:
            # Analisi base esistente
            dow_stats = df.groupby('Day_of_Week').agg({
                'Volume': 'mean',
                'Change_Pct': lambda x: abs(x).mean(),
                'Daily_Range': 'mean'
            }).round(4)
            
            # Nuove analisi dettagliate per giorno della settimana
            dow_detailed = {}
            for day in df['Day_of_Week'].unique():
                day_data = df[df['Day_of_Week'] == day]
                positive_data = day_data[day_data['Change_Pct'] > 0]
                negative_data = day_data[day_data['Change_Pct'] < 0]
                
                dow_detailed[day] = {
                    "avg_change_overall": round(float(day_data['Change_Pct'].mean()), 4),
                    "avg_change_positive": round(float(positive_data['Change_Pct'].mean()), 4) if len(positive_data) > 0 else 0.0,
                    "avg_change_negative": round(float(negative_data['Change_Pct'].mean()), 4) if len(negative_data) > 0 else 0.0,
                    "total_days": len(day_data),
                    "positive_days": len(positive_data),
                    "negative_days": len(negative_data),
                    "positive_percentage": round(len(positive_data) / len(day_data) * 100, 2) if len(day_data) > 0 else 0.0,
                    "negative_percentage": round(len(negative_data) / len(day_data) * 100, 2) if len(day_data) > 0 else 0.0
                }
            
            if len(dow_stats) > 0:
                result["day_of_week"] = {
                    "volume_max": {
                        "day": dow_stats['Volume'].idxmax(),
                        "value": round(float(dow_stats['Volume'].max()), 4)
                    },
                    "volume_min": {
                        "day": dow_stats['Volume'].idxmin(),
                        "value": round(float(dow_stats['Volume'].min()), 4)
                    },
                    "change_max": {
                        "day": dow_stats['Change_Pct'].idxmax(),
                        "value": round(float(dow_stats['Change_Pct'].max()), 4)
                    },
                    "change_min": {
                        "day": dow_stats['Change_Pct'].idxmin(),
                        "value": round(float(dow_stats['Change_Pct'].min()), 4)
                    },
                    "range_max": {
                        "day": dow_stats['Daily_Range'].idxmax(),
                        "value": round(float(dow_stats['Daily_Range'].max()), 4)
                    },
                    "range_min": {
                        "day": dow_stats['Daily_Range'].idxmin(),
                        "value": round(float(dow_stats['Daily_Range'].min()), 4)
                    },
                    "detailed_analysis": dow_detailed
                }
            else:
                result["day_of_week"] = self._empty_seasonal_section()
        else:
            result["day_of_week"] = self._empty_seasonal_section()
        
        # Week of Month analysis
        if 'Week_of_Month' in df.columns and len(df) > 0:
            # Analisi base esistente
            wom_stats = df.groupby('Week_of_Month').agg({
                'Volume': 'mean',
                'Change_Pct': lambda x: abs(x).mean(),
                'Daily_Range': 'mean'
            }).round(4)
            
            # Nuove analisi dettagliate per settimana del mese
            wom_detailed = {}
            for week in df['Week_of_Month'].unique():
                week_data = df[df['Week_of_Month'] == week]
                positive_data = week_data[week_data['Change_Pct'] > 0]
                negative_data = week_data[week_data['Change_Pct'] < 0]
                
                wom_detailed[int(week)] = {
                    "avg_change_overall": round(float(week_data['Change_Pct'].mean()), 4),
                    "avg_change_positive": round(float(positive_data['Change_Pct'].mean()), 4) if len(positive_data) > 0 else 0.0,
                    "avg_change_negative": round(float(negative_data['Change_Pct'].mean()), 4) if len(negative_data) > 0 else 0.0,
                    "total_days": len(week_data),
                    "positive_days": len(positive_data),
                    "negative_days": len(negative_data),
                    "positive_percentage": round(len(positive_data) / len(week_data) * 100, 2) if len(week_data) > 0 else 0.0,
                    "negative_percentage": round(len(negative_data) / len(week_data) * 100, 2) if len(week_data) > 0 else 0.0
                }
            
            if len(wom_stats) > 0:
                result["week_of_month"] = {
                    "volume_max": {
                        "week": int(wom_stats['Volume'].idxmax()),
                        "value": round(float(wom_stats['Volume'].max()), 4)
                    },
                    "volume_min": {
                        "week": int(wom_stats['Volume'].idxmin()),
                        "value": round(float(wom_stats['Volume'].min()), 4)
                    },
                    "change_max": {
                        "week": int(wom_stats['Change_Pct'].idxmax()),
                        "value": round(float(wom_stats['Change_Pct'].max()), 4)
                    },
                    "change_min": {
                        "week": int(wom_stats['Change_Pct'].idxmin()),
                        "value": round(float(wom_stats['Change_Pct'].min()), 4)
                    },
                    "range_max": {
                        "week": int(wom_stats['Daily_Range'].idxmax()),
                        "value": round(float(wom_stats['Daily_Range'].max()), 4)
                    },
                    "range_min": {
                        "week": int(wom_stats['Daily_Range'].idxmin()),
                        "value": round(float(wom_stats['Daily_Range'].min()), 4)
                    },
                    "detailed_analysis": wom_detailed
                }
            else:
                result["week_of_month"] = self._empty_seasonal_section_week()
        else:
            result["week_of_month"] = self._empty_seasonal_section_week()
        
        # Month analysis
        if 'Month' in df.columns and len(df) > 0:
            # Analisi base esistente
            month_stats = df.groupby('Month').agg({
                'Volume': 'mean',
                'Change_Pct': lambda x: abs(x).mean(),
                'Daily_Range': 'mean'
            }).round(4)
            
            # Nuove analisi dettagliate per mese
            month_detailed = {}
            for month in df['Month'].unique():
                month_data = df[df['Month'] == month]
                positive_data = month_data[month_data['Change_Pct'] > 0]
                negative_data = month_data[month_data['Change_Pct'] < 0]
                
                month_detailed[int(month)] = {
                    "avg_change_overall": round(float(month_data['Change_Pct'].mean()), 4),
                    "avg_change_positive": round(float(positive_data['Change_Pct'].mean()), 4) if len(positive_data) > 0 else 0.0,
                    "avg_change_negative": round(float(negative_data['Change_Pct'].mean()), 4) if len(negative_data) > 0 else 0.0,
                    "total_days": len(month_data),
                    "positive_days": len(positive_data),
                    "negative_days": len(negative_data),
                    "positive_percentage": round(len(positive_data) / len(month_data) * 100, 2) if len(month_data) > 0 else 0.0,
                    "negative_percentage": round(len(negative_data) / len(month_data) * 100, 2) if len(month_data) > 0 else 0.0
                }
            
            if len(month_stats) > 0:
                result["month"] = {
                    "volume_max": {
                        "month": int(month_stats['Volume'].idxmax()),
                        "value": round(float(month_stats['Volume'].max()), 4)
                    },
                    "volume_min": {
                        "month": int(month_stats['Volume'].idxmin()),
                        "value": round(float(month_stats['Volume'].min()), 4)
                    },
                    "change_max": {
                        "month": int(month_stats['Change_Pct'].idxmax()),
                        "value": round(float(month_stats['Change_Pct'].max()), 4)
                    },
                    "change_min": {
                        "month": int(month_stats['Change_Pct'].idxmin()),
                        "value": round(float(month_stats['Change_Pct'].min()), 4)
                    },
                    "range_max": {
                        "month": int(month_stats['Daily_Range'].idxmax()),
                        "value": round(float(month_stats['Daily_Range'].max()), 4)
                    },
                    "range_min": {
                        "month": int(month_stats['Daily_Range'].idxmin()),
                        "value": round(float(month_stats['Daily_Range'].min()), 4)
                    },
                    "detailed_analysis": month_detailed
                }
            else:
                result["month"] = self._empty_seasonal_section_month()
        else:
            result["month"] = self._empty_seasonal_section_month()
        
        # Aggiungi analisi globale per giorni positivi/negativi
        positive_days_total = len(df[df['Change_Pct'] > 0])
        negative_days_total = len(df[df['Change_Pct'] < 0])
        total_days = len(df)
        
        result["global_summary"] = {
            "total_days": total_days,
            "positive_days_total": positive_days_total,
            "negative_days_total": negative_days_total,
            "positive_percentage_total": round(positive_days_total / total_days * 100, 2) if total_days > 0 else 0.0,
            "negative_percentage_total": round(negative_days_total / total_days * 100, 2) if total_days > 0 else 0.0
        }
        
        # Dettaglio giorni positivi per giorno della settimana
        if 'Day_of_Week' in df.columns:
            positive_by_dow = {}
            negative_by_dow = {}
            
            for day in df['Day_of_Week'].unique():
                day_positive = len(df[(df['Day_of_Week'] == day) & (df['Change_Pct'] > 0)])
                day_negative = len(df[(df['Day_of_Week'] == day) & (df['Change_Pct'] < 0)])
                
                positive_by_dow[day] = {
                    "count": day_positive,
                    "percentage_of_total_positive": round(day_positive / positive_days_total * 100, 2) if positive_days_total > 0 else 0.0,
                    "percentage_of_day_total": round(day_positive / len(df[df['Day_of_Week'] == day]) * 100, 2) if len(df[df['Day_of_Week'] == day]) > 0 else 0.0
                }
                
                negative_by_dow[day] = {
                    "count": day_negative,
                    "percentage_of_total_negative": round(day_negative / negative_days_total * 100, 2) if negative_days_total > 0 else 0.0,
                    "percentage_of_day_total": round(day_negative / len(df[df['Day_of_Week'] == day]) * 100, 2) if len(df[df['Day_of_Week'] == day]) > 0 else 0.0
                }
            
            result["global_summary"]["positive_days_by_weekday"] = positive_by_dow
            result["global_summary"]["negative_days_by_weekday"] = negative_by_dow
        
        return result
    
    def _empty_seasonal_patterns(self):
        """Ritorna struttura vuota per seasonal_patterns"""
        return {
            "day_of_week": self._empty_seasonal_section(),
            "week_of_month": self._empty_seasonal_section_week(),
            "month": self._empty_seasonal_section_month(),
            "global_summary": {
                "total_days": 0,
                "positive_days_total": 0,
                "negative_days_total": 0,
                "positive_percentage_total": 0.0,
                "negative_percentage_total": 0.0,
                "positive_days_by_weekday": {},
                "negative_days_by_weekday": {}
            }
        }
    
    def _empty_seasonal_section(self):
        return {
            "volume_max": {"day": "N/A", "value": 0.0},
            "volume_min": {"day": "N/A", "value": 0.0},
            "change_max": {"day": "N/A", "value": 0.0},
            "change_min": {"day": "N/A", "value": 0.0},
            "range_max": {"day": "N/A", "value": 0.0},
            "range_min": {"day": "N/A", "value": 0.0},
            "detailed_analysis": {}
        }
    
    def _empty_seasonal_section_week(self):
        return {
            "volume_max": {"week": 0, "value": 0.0},
            "volume_min": {"week": 0, "value": 0.0},
            "change_max": {"week": 0, "value": 0.0},
            "change_min": {"week": 0, "value": 0.0},
            "range_max": {"week": 0, "value": 0.0},
            "range_min": {"week": 0, "value": 0.0},
            "detailed_analysis": {}
        }
    
    def _empty_seasonal_section_month(self):
        return {
            "volume_max": {"month": 0, "value": 0.0},
            "volume_min": {"month": 0, "value": 0.0},
            "change_max": {"month": 0, "value": 0.0},
            "change_min": {"month": 0, "value": 0.0},
            "range_max": {"month": 0, "value": 0.0},
            "range_min": {"month": 0, "value": 0.0},
            "detailed_analysis": {}
        }
    
    def analyze_bollinger_volatility(self, df):
        """Analizza Bollinger Bands e volatilità"""
        if len(df) == 0:
            return self._empty_bollinger_volatility()
            
        result = {}
        
        # BB Squeeze analysis
        squeeze_count = int(df['BB_Squeeze'].sum())
        total_days = len(df)
        squeeze_percentage = round(squeeze_count / total_days * 100, 2) if total_days > 0 else 0
        
        result["bb_squeeze_total"] = {
            "count": squeeze_count,
            "percentage": squeeze_percentage
        }
        
        # BB Width analysis
        result["bb_width_top5_max"] = self.get_top5_with_details(df, 'BB_Width', ascending=False)
        result["bb_width_top5_min"] = self.get_top5_with_details(df, 'BB_Width', ascending=True)
        result["bb_width_max"] = self.get_extreme_with_details(df, 'BB_Width', is_max=True)
        result["bb_width_min"] = self.get_extreme_with_details(df, 'BB_Width', is_max=False)
        
        # ATR analysis
        result["atr_top5_max"] = self.get_top5_with_details(df, 'ATR_14', ascending=False)
        result["atr_top5_min"] = self.get_top5_with_details(df, 'ATR_14', ascending=True)
        result["atr_max"] = self.get_extreme_with_details(df, 'ATR_14', is_max=True)
        result["atr_min"] = self.get_extreme_with_details(df, 'ATR_14', is_max=False)
        result["atr_avg"] = round(float(df['ATR_14'].mean()), 4)
        
        # BB External Distance (quando il prezzo è fuori dalle bande)
        df_copy = df.copy()
        df_copy['bb_external_distance'] = 0.0
        
        # Calcola distanza quando il prezzo è fuori dalle bande
        above_upper = df_copy['Close'] > df_copy['BB_Upper']
        below_lower = df_copy['Close'] < df_copy['BB_Lower']
        
        df_copy.loc[above_upper, 'bb_external_distance'] = (
            df_copy.loc[above_upper, 'Close'] - df_copy.loc[above_upper, 'BB_Upper']
        ) / df_copy.loc[above_upper, 'Close'] * 100
        
        df_copy.loc[below_lower, 'bb_external_distance'] = (
            df_copy.loc[below_lower, 'BB_Lower'] - df_copy.loc[below_lower, 'Close']
        ) / df_copy.loc[below_lower, 'Close'] * 100
        
        # Solo i giorni con distanza > 0 (fuori dalle bande)
        external_df = df_copy[df_copy['bb_external_distance'] > 0]
        
        if len(external_df) > 0:
            result["bb_external_distance_top5"] = self.get_top5_with_details(
                external_df, 'bb_external_distance', ascending=False
            )
            result["bb_external_distance_max"] = self.get_extreme_with_details(
                external_df, 'bb_external_distance', is_max=True
            )
        else:
            result["bb_external_distance_top5"] = []
            result["bb_external_distance_max"] = {"date": "N/A", "value": 0.0}
        
        return result
    
    def _empty_bollinger_volatility(self):
        """Ritorna struttura vuota per bollinger_volatility"""
        empty_detail = {"date": "N/A", "value": 0.0}
        return {
            "bb_squeeze_total": {"count": 0, "percentage": 0.0},
            "bb_width_top5_max": [],
            "bb_width_top5_min": [],
            "bb_width_max": empty_detail,
            "bb_width_min": empty_detail,
            "atr_top5_max": [],
            "atr_top5_min": [],
            "atr_max": empty_detail,
            "atr_min": empty_detail,
            "atr_avg": 0.0,
            "bb_external_distance_top5": [],
            "bb_external_distance_max": empty_detail
        }
    
    def analyze_single_timeframe(self, df, period_name):
        """Esegue l'analisi completa per un singolo timeframe"""
        if len(df) == 0:
            print(f"   ⚠️  No data for {period_name}")
            return None
        
        # Esegue tutte le analisi
        result = {
            "period": period_name,
            "data_quality": self.get_data_quality(df),
            "core_analysis": {
                "basic_performance": self.analyze_basic_performance(df),
                "volume_range_extremes": self.analyze_volume_range_extremes(df),
                "ma_analysis": self.analyze_ma_analysis(df),
                "seasonal_patterns": self.analyze_seasonal_patterns(df),
                "bollinger_volatility": self.analyze_bollinger_volatility(df)
            }
        }
        
        return result
    
    def generate_all_analyses(self):
        """Genera automaticamente tutte le analisi per tutti i timeframe"""
        results = {}
        
        print("\n🚀 STARTING COMPREHENSIVE ANALYSIS")
        print("=" * 50)
        
        # 1. Analisi TOTAL (tutto il dataset)
        print("📊 Analyzing TOTAL dataset...")
        total_result = self.analyze_single_timeframe(self.df, "total")
        if total_result:
            results["total"] = total_result
            print(f"   ✅ Total: {len(self.df)} days analyzed")
        
        # 2. Analisi per ogni ANNO
        print("\n📅 Analyzing YEARS...")
        results["years"] = {}
        
        for year in sorted(self.timeframes['years'].keys()):
            year_df = self.df[self.df['Date'].dt.year == year]
            year_result = self.analyze_single_timeframe(year_df, str(year))
            
            if year_result:
                results["years"][year] = {
                    "analysis": year_result,
                    "quarters": {}
                }
                print(f"   ✅ {year}: {len(year_df)} days analyzed")
                
                # 3. Analisi per ogni TRIMESTRE dell'anno
                quarters_info = self.timeframes['years'][year]['quarters']
                for quarter_num, quarter_info in quarters_info.items():
                    quarter_period = quarter_info['period_code']  # Es: "2020-Q1"
                    quarter_df = year_df[year_df['Quarter'] == quarter_period]
                    quarter_result = self.analyze_single_timeframe(quarter_df, quarter_period)
                    
                    if quarter_result:
                        results["years"][year]["quarters"][quarter_num] = quarter_result
                        print(f"     ✅ {quarter_period}: {len(quarter_df)} days analyzed")
        
        print(f"\n🎯 ANALYSIS COMPLETED")
        print(f"   - Total timeframes analyzed: {self._count_analyses(results)}")
        
        return results
    
    def _count_analyses(self, results):
        """Conta il numero totale di analisi generate"""
        count = 0
        if "total" in results:
            count += 1
        
        if "years" in results:
            for year_data in results["years"].values():
                count += 1  # Anno
                count += len(year_data["quarters"])  # Trimestri
        
        return count
    
    def save_all_results(self, results, output_dir="analysis_output"):
        """Salva tutti i risultati nella struttura gerarchica"""
        base_path = Path(output_dir)
        
        print(f"\n💾 SAVING RESULTS TO: {base_path}")
        print("=" * 50)
        
        saved_files = []
        
        # 1. Salva TOTAL
        if "total" in results:
            total_path = base_path / "total_program1.json"
            total_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(total_path, 'w', encoding='utf-8') as f:
                json.dump(results["total"], f, indent=2, ensure_ascii=False)
            
            saved_files.append(str(total_path))
            print(f"📄 {total_path}")
        
        # 2. Salva ANNI e TRIMESTRI
        if "years" in results:
            for year, year_data in results["years"].items():
                year_str = str(year)
                year_dir = base_path / year_str
                
                # Salva analisi dell'anno
                if "analysis" in year_data:
                    year_path = year_dir / f"{year_str}_program1.json"
                    year_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(year_path, 'w', encoding='utf-8') as f:
                        json.dump(year_data["analysis"], f, indent=2, ensure_ascii=False)
                    
                    saved_files.append(str(year_path))
                    print(f"📁 {year_path}")
                
                # Salva trimestri
                if "quarters" in year_data:
                    for quarter_num, quarter_result in year_data["quarters"].items():
                        quarter_dir = year_dir / f"Q{quarter_num}"
                        quarter_path = quarter_dir / f"{year_str}-Q{quarter_num}_program1.json"
                        quarter_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(quarter_path, 'w', encoding='utf-8') as f:
                            json.dump(quarter_result, f, indent=2, ensure_ascii=False)
                        
                        saved_files.append(str(quarter_path))
                        print(f"   📄 {quarter_path}")
        
        print(f"\n✅ SAVED {len(saved_files)} FILES")
        return saved_files
    
    def run_complete_analysis(self, output_dir="analysis_output"):
        """Esegue l'analisi completa automatica"""
        print("🎯 CORE OIL ANALYZER - COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        # Genera tutte le analisi
        all_results = self.generate_all_analyses()
        
        # Salva tutti i risultati
        saved_files = self.save_all_results(all_results, output_dir)
        
        # Summary finale
        print(f"\n🏆 ANALYSIS SUMMARY")
        print("=" * 30)
        print(f"📊 Dataset: {len(self.df)} total records")
        print(f"📅 Period: {self.df['Date'].min().strftime('%Y-%m-%d')} to {self.df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Files generated: {len(saved_files)}")
        print(f"🎯 Analysis types: Basic Performance, Volume Extremes, MA Analysis, Seasonal Patterns, Bollinger Volatility")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description='Core Oil Analyzer - Program 1: Complete Automatic Analysis')
    parser.add_argument('--input', default='/Users/e/Desktop/oilProva/oil_processed.csv', help='Input CSV file path (oil_processed.csv)')
    parser.add_argument('--output', default='/Users/e/Desktop/oilProva/oil_analysis', help='Output directory for hierarchical structure')
    
    args = parser.parse_args()
    
    try:
        # Inizializza l'analyzer
        analyzer = CoreOilAnalyzer(args.input)
        
        # Esegue l'analisi completa automatica
        results = analyzer.run_complete_analysis(args.output)
        
        print(f"\n🎉 SUCCESS! Core Oil Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()
