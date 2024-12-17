

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from prettytable import PrettyTable
from sklearn.cluster import KMeans, DBSCAN
from tabulate import tabulate
# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import xgboost as xgb
import statsmodels.api as sm

from scipy.linalg import svd
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from numpy.linalg import svd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Tuple, Union, Optional, Dict
# Visualization
from prettytable import PrettyTable
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_curve, auc, classification_report)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
class PowerliftingDataProcessor:
    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)
        self.federation_tiers = self._define_federation_tiers()
        self.train_data = None
        self.test_data = None
        self.equipment_advantages = None

    def _check_and_report_nans(self, data: pd.DataFrame, name: str = ""):
        """Check for NaN values in each column and report percentage"""
        nan_info = {}
        for column in data.columns:
            nan_count = data[column].isna().sum()
            nan_percentage = (nan_count / len(data)) * 100
            if nan_count > 0:
                nan_info[column] = {
                    'count': nan_count,
                    'percentage': nan_percentage
                }
        return nan_info

    def visualization_Equipment_Total(self):
        equipment_types = ['Raw', 'Wraps', 'Single-ply', 'Multi-ply']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))


        for i, equipment in enumerate(equipment_types):
            row, col = divmod(i, 2)
            data = self.df[self.df['Equipment'] == equipment]

            stats = data.groupby('Sex')['Strength'].agg(['mean', 'std']).round(2)
            sns.histplot(data=data, x='Strength', hue='Sex', ax=axes[row, col])
            axes[row, col].set_xlabel('Strength (Total/Bodyweight)')
            stats_text = '\n'.join([
                f"{sex} - Mean: {stats.loc[sex, 'mean']:.2f}, STD: {stats.loc[sex, 'std']:.2f}"
                for sex in stats.index
            ])
            axes[row, col].set_title(f'{equipment} Strength Distribution\n{stats_text}', pad=20)

        plt.tight_layout()

        plt.show()

    def visualization_Tested_Strength_boxplot(self):
        df = self.df.copy()
        df['WeightBin'] = pd.cut(df['BodyweightKg'],
                                 bins=np.arange(40, 160, 10),
                                 labels=[f'{i}-{i + 10}kg' for i in range(40, 150, 10)])

        # Create subplots for each sex
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))  # 2 rows (M/F) x 4 equipment types

        # Get unique equipment types
        equipment_types = df['Equipment'].unique()

        # Plot for each sex
        for sex_idx, sex in enumerate(['M', 'F']):
            for eq_idx, equip in enumerate(equipment_types):
                # Filter data for current sex and equipment
                plot_data = df[(df['Sex'] == sex) & (df['Equipment'] == equip)]

                # Create boxplot
                sns.scatterplot(data=plot_data,
                            x='WeightBin',
                            y='Strength',
                            hue='Tested',
                            alpha= .3,
                            ax=axes[sex_idx, eq_idx])

                # Customize plot
                axes[sex_idx, eq_idx].set_title(f'{equip} - {sex}')
                axes[sex_idx, eq_idx].set_xticklabels(axes[sex_idx, eq_idx].get_xticklabels(), rotation=45)
                axes[sex_idx, eq_idx].set_xlabel('Weight Bins (kg)')
                axes[sex_idx, eq_idx].set_ylabel('Relative Strength')

        plt.suptitle('Strength Distribution by Sex, Equipment, Weight Class, and Testing Status', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()


        '''
        # Print summary statistics
        for sex in ['M', 'F']:
            for equip in equipment_types:
                print(f"\n{equip} Equipment - {sex}:")
                summary = df[(df['Sex'] == sex) & (df['Equipment'] == equip)].groupby(['WeightBin', 'Tested'])[
                    'Strength'].describe()
                print(summary)
       '''

    def visulization_Age_Histogram(self,col,bins):
        fig, axes = plt.subplots(5, 2, figsize=(15, 12))

        # Create age bins (every 5 years from 18 to 60)


        age_bins = np.arange(self.df[col].min(), self.df[col].max(), bins)
        # Raw
        data_raw = self.df[self.df['Equipment'] == 'Raw']
        sns.histplot(data=data_raw[data_raw['Sex'] == 'M'], x=col, bins=age_bins,
                     ax=axes[0, 0], color='blue', alpha=0.5, label='Male')
        sns.histplot(data=data_raw[data_raw['Sex'] == 'F'], x=col, bins=age_bins,
                     ax=axes[0, 1], color='orange', alpha=0.5, label='Female')
        axes[0, 0].set_title("Raw "+ col +" Male Distribution")
        axes[0, 1].set_title("Raw " + col+ " Female Distribution")


        # Wraps
        data_wraps = self.df[self.df['Equipment'] == 'Wraps']
        sns.histplot(data=data_wraps[data_wraps['Sex'] == 'M'], x=col, bins=age_bins,
                     ax=axes[1, 0], color='blue', alpha=0.5, label='Male')
        sns.histplot(data=data_wraps[data_wraps['Sex'] == 'F'], x=col, bins=age_bins,
                     ax=axes[1, 1], color='orange', alpha=0.5, label='Female')
        axes[1, 0].set_title("Wraps  "+ col + " Male Distribution")
        axes[1, 1].set_title("Wraps  "+ col  +  " Female Distribution")


        # Single-ply
        data_single = self.df[self.df['Equipment'] == 'Single-ply']
        sns.histplot(data=data_single[data_single['Sex'] == 'M'], x=col, bins=age_bins,
                     ax=axes[2, 0], color='blue', alpha=0.5, label='Male')
        sns.histplot(data=data_single[data_single['Sex'] == 'F'], x=col, bins=age_bins,
                     ax=axes[2, 1], color='orange', alpha=0.5, label='Female')
        axes[2, 0].set_title("Single-ply "+ col +" Male Distribution")
        axes[2, 1].set_title("Single-ply "+ col +" Female Distribution")
        axes[1, 0].legend()

        data_multi = self.df[self.df['Equipment'] == 'Multi-ply']
        sns.histplot(data=data_multi[data_multi['Sex'] == 'M'], x=col, bins=age_bins,
                     ax=axes[3, 0], color='blue', alpha=0.5, label='Male')
        sns.histplot(data=data_multi[data_multi['Sex'] == 'F'], x=col, bins=age_bins,
                     ax=axes[3, 1], color='orange', alpha=0.5, label='Female')
        axes[3, 0].set_title("Multi-ply "+ col +" Male Distribution")
        axes[3, 1].set_title("Multi-ply "+ col +" Female Distribution")


        sns.histplot(data=self.df[self.df['Sex'] == 'M'], x=col, bins=age_bins,
                     ax=axes[4, 0], color='blue', alpha=0.5, label='Male')
        sns.histplot(data=self.df[self.df['Sex'] == 'F'], x=col, bins=age_bins,
                     ax=axes[4, 1], color='orange', alpha=0.5, label='Female')
        axes[4, 0].set_title("Equipment independent " +col + "Male Distribution")
        axes[4, 1].set_title("Equipment independent " +col +"Female Distribution")


        # Common labels and adjustments
        for row in axes:
            for ax in row:
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.set_xticks(age_bins)
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def visulization_Age_Strength(self):
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))

        sns.lineplot(data=self.df[self.df['Equipment']=='Raw'],y='Strength',x='Age',hue='Sex',ax=axes[0, 0])
        axes[0,0].set_title("Raw Strength Distributuion vs Age")
        sns.lineplot(data=self.df[self.df['Equipment']=='Wraps'],y='Strength',x='Age',hue='Sex',ax=axes[0, 1])
        axes[0,1].set_title("Wraps Strength Distributuion vs Age")

        sns.lineplot(data=self.df[self.df['Equipment']=='Single-ply'],y='Strength',x='Age',hue='Sex',ax=axes[1, 0])
        axes[1,0].set_title("Single-ply Strength Distributuion vs Age")

        sns.lineplot(data=self.df[self.df['Equipment']=='Multi-ply'],y='Strength',x='Age',hue='Sex',ax=axes[1, 1])
        axes[1, 1].set_title("Multi Strength Distributuion vs Age")

        sns.scatterplot(data=self.df[(self.df['Equipment'] == 'Raw') & (self.df['Sex'] == 'M') ] , y='Strength', x='Age', ax=axes[2, 0])
        axes[2, 0].set_title("Raw Strength Distributuion vs Age ScatterPlot ")

        sns.scatterplot(data=self.df[(self.df['Equipment'] == 'Wraps')  & (self.df['Sex'] == 'M') ], y='Strength', x='Age', ax=axes[2, 1])
        axes[2, 1].set_title("Wraps Strength Distributuion vs Age ScatterPlot ")

        sns.scatterplot(data=self.df[(self.df['Equipment'] == 'Single-ply') & (self.df['Sex'] == 'M') ], y='Strength', x='Age',ax=axes[3, 0])
        axes[3, 0].set_title("Single-ply' Strength Distributuion vs Age ScatterPlot ")

        sns.scatterplot(data=self.df[(self.df['Equipment'] == 'Multi-ply') & (self.df['Sex'] == 'M') ], y='Strength', x='Age', ax=axes[3, 1])
        axes[3, 0].set_title("Multi-ply' Strength Distributuion vs Age ScatterPlot ")

        plt.tight_layout()

        plt.show()

    def visulization_scatter_col_Strength(self,col):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        sns.scatterplot(data=self.df[self.df['Equipment'] == 'Raw'], y='Strength', x=col, hue='Sex', ax=axes[0, 0],alpha=.4)
        axes[0, 0].set_title("Raw Strength Distributuion vs "+col)
        sns.scatterplot(data=self.df[self.df['Equipment'] == 'Wraps'], y='Strength', x=col, hue='Sex', ax=axes[0, 1],alpha=.4)
        axes[0, 1].set_title("Wraps Strength Distributuion vs "+col)

        sns.scatterplot(data=self.df[self.df['Equipment'] == 'Single-ply'], y='Strength', x=col, hue='Sex',
                     ax=axes[1, 0],alpha=.4)
        axes[1, 0].set_title("Single-ply Strength Distributuion vs "+col)

        sns.scatterplot(data=self.df[self.df['Equipment'] == 'Multi-ply'], y='Strength', x=col, hue='Sex', ax=axes[1, 1],alpha=.4)
        axes[1, 1].set_title("Multi Strength Distributuion vs "+col)



        plt.tight_layout()

        plt.show()

    def remove_bodyweight_outliers(self):
        df_clean = self.df.copy()

        # Handle each equipment and sex combination separately
        for equipment in df_clean['Equipment'].unique():
            for sex in df_clean['Sex'].unique():
                mask = (df_clean['Equipment'] == equipment) & (df_clean['Sex'] == sex)

                # Calculate IQR
                Q1 = df_clean[mask]['BodyweightKg'].quantile(0.25)
                Q3 = df_clean[mask]['BodyweightKg'].quantile(0.75)
                IQR = Q3 - Q1

                # Use 2.0 instead of 1.5 to be more conservative
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Remove outliers
                df_clean = df_clean[~(mask & ((df_clean['BodyweightKg'] < lower_bound) |
                                              (df_clean['BodyweightKg'] > upper_bound)))]

        return df_clean


    def process_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Main data processing pipeline with train-test split"""


        self._handle_missing_values()
        self._remove_unnecessary_columns()
        self._clean_lift_data()

        # Create basic features before split
        self.df['Strength'] = self.df['TotalKg'] / self.df['BodyweightKg']
        self.visulization_Age_Strength()
        self.visulization_scatter_col_Strength('BodyweightKg')
        self.df['SquatStrength'] = self.df['Best3SquatKg'] / self.df['BodyweightKg']
        self.visualization_Tested_Strength_boxplot()

        self.df=self.remove_bodyweight_outliers()
        self.visulization_Age_Histogram("BodyweightKg",10)


        self._remove_outliers()
        self.visulization_Age_Histogram("SquatStrength",.5)


        self.visualization_Equipment_Total()
        self.df['Federation_Tier'] = self.df['Federation'].apply(self._get_federation_tier)

        # Split data before calculating equipment advantage
        self.train_data, self.test_data = train_test_split(
            self.df, train_size=0.8, random_state=42,stratify=self.df['Tested']
        )

        """Calculate equipment advantage using only training data Not using Test Data to avoid 
        # Data Leakage"""
        self._calculate_equipment_advantage()



        #Final Check for NAN values in processed dataset
        print("\nNaN values in Training Set:")
        train_nans = self._check_and_report_nans(self.train_data)
        for col, info in train_nans.items():
            print(f"{col}: {info['count']} NaNs ({info['percentage']:.2f}%)")

        print("\nNaN values in Test Set:")
        test_nans = self._check_and_report_nans(self.test_data)
        for col, info in test_nans.items():
            print(f"{col}: {info['count']} NaNs ({info['percentage']:.2f}%)")


        return self.train_data, self.test_data

    def _remove_unnecessary_columns(self):
        """Remove unnecessary columns from dataset"""
        unnecessary_columns = [
            'Name', 'Squat1Kg', 'Squat2Kg', 'Squat3Kg', 'Squat4Kg',
            'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 'Bench4Kg', 'Deadlift1Kg',
            'Deadlift2Kg', 'Deadlift3Kg', 'Deadlift4Kg', 'MeetState',
             'Wilks', 'McCulloch', 'Glossbrenner', 'Place','MeetName',
            'IPFPoints','AgeClass','WeightClassKg','Country','Event','Division'
        ]
        self.df.drop(unnecessary_columns, axis=1, inplace=True)

    def _histograms(self, col, target):
        """Plot histogram of target variable grouped by column"""
        # Option 1: Single histogram for all countries
        sns.histplot(
            data=self.df,
            x=target,
            hue=col,  # This will color-code by country
            multiple="stack"  # or "layer" for overlapping
        )
        plt.title(f'Distribution of {target} by {col}')
        plt.xlabel(target)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()




    def _remove_outliers(self):
        """Remove outliers from the dataset"""
        # Remove bodyweight outliers
       # self.df = self._remove_outliers_by_zscore('BodyweightKg', 2)

        # Remove age outliers and filter minimum age
        self.df = self._remove_outliers_by_zscore('Age', 3)
        self.df = self.df[self.df['Age'] > 16]

        # Remove strength outliers by sex
        self.df = self._remove_lower_outliers_by_groups('Strength',-3)
        self.df = self._remove_lower_outliers_by_groups('SquatStrength',-2)


    def _remove_outliers_by_zscore(self, column: str, threshold: float) -> pd.DataFrame:
        """Remove outliers based on z-score"""
        z = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        return self.df[z < threshold]

    def _remove_lower_outliers_by_groups(self, column: str,threshhold) -> pd.DataFrame:
        """Remove lower outliers separately for each sex and equipment type"""
        df_clean = self.df.copy()

        # Group by sex and equipment
        for (sex, equipment) in df_clean.groupby(['Sex', 'Equipment']).groups:
            # Create mask for current group
            mask = ((df_clean['Sex'] == sex) &
                    (df_clean['Equipment'] == equipment))

            # Calculate statistics for current group
            group_data = df_clean[mask][column]
            if len(group_data) > 0:  # Check if group has data
                group_mean = group_data.mean()
                group_std = group_data.std()

                # Calculate z-scores
                z = (group_data - group_mean) / group_std

                # Apply threshold
                threshold_z = threshhold
                df_clean = df_clean[~(mask & (z <= threshold_z))]

        return df_clean

    @staticmethod
    def _define_federation_tiers() -> dict:
        """Define federation tiers"""
        return {
            'ELITE': ['IPF', 'USAPL', 'EPF', 'AsianPF', 'OceaniaPF', 'NAPF', 'CPU'],
            'MAJOR': ['USPA', 'FPR', 'UkrainePF', 'CSST', 'PA', 'APF', 'RPS'],
            'REGIONAL': ['UPA', 'USPF', 'IPL', 'NASA', 'IPA', 'WPC-RUS', 'BPU']
        }

    def _get_federation_tier(self, federation: str) -> str:
        """Get federation tier for a given federation"""
        for tier, feds in self.federation_tiers.items():
            if federation in feds:
                return tier
        return 'LOCAL'

    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        self.df['Tested'] = self.df['Tested'].map({'Yes': 1}).fillna(0)
        self.df['Year'] = pd.to_datetime(self.df['Date']).dt.year

        """Handle missing values of Age in the dataset"""
        cleanDf =  self.df[~self.df['Age'].isna()]
        avg_age_by_division = cleanDf.groupby('Division')['Age'].mean()

        def get_age_from_division(division):
            if division in avg_age_by_division:
                return avg_age_by_division[division]

            return np.nan

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        missing_age_mask = self.df['Age'].isna()
        self.df.loc[missing_age_mask, 'Age'] = self.df.loc[missing_age_mask, 'Division'].apply(get_age_from_division)
        self.df.dropna(subset=['TotalKg', 'Best3DeadliftKg', 'Best3BenchKg',
                               'Best3SquatKg','BodyweightKg','Age'], inplace=True)

    def _clean_lift_data(self):
        """Clean and validate lift data"""
        self.df = self.df[self.df['Best3SquatKg'] > 0]
        self.df = self.df[self.df['Best3BenchKg'] > 0]
        self.df = self.df[self.df['Best3DeadliftKg'] > 0]
        self.df = self.df[self.df['TotalKg'] >= self.df['Best3SquatKg']]
        self.df = self.df[self.df['TotalKg'] >= self.df['Best3BenchKg']]
        self.df = self.df[self.df['TotalKg'] >= self.df['Best3DeadliftKg']]

    def _calculate_equipment_advantage(self):
        """Calculate equipment advantage using only training data"""
        # Calculate baselines from training data
        baselines = {
            sex: self.train_data[
                (self.train_data['Equipment'] == 'Raw') &
                (self.train_data['Sex'] == sex)
                ]['Strength'].mean()
            for sex in ['M', 'F']
        }

        self.equipment_advantages = {}
        for sex in ['M', 'F']:
            baseline = baselines[sex]
            for equip in self.train_data['Equipment'].unique():
                mask = (self.train_data['Sex'] == sex) & (self.train_data['Equipment'] == equip)
                avg_strength = self.train_data[mask]['Strength'].mean()
                self.equipment_advantages[(sex, equip)] = (avg_strength - baseline) / baseline

        # Apply to both train and test data
        for data in [self.train_data, self.test_data]:
            data['Equipment_Advantage'] = data.apply(
                lambda row: self.equipment_advantages[(row['Sex'], row['Equipment'])],
                axis=1
            )




class PrepareFeatures:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        self.train_data = train_data
        self.test_data = test_data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_features_regression(self):
        """Prepare features for modeling"""
        le = LabelEncoder()
        self.train_data['Sex_encoded'] = le.fit_transform(self.train_data['Sex'])
        self.test_data['Sex_encoded'] = le.transform(self.test_data['Sex'])

        # Create dummies for federation tiers
        train_federation_dummies = pd.get_dummies(self.train_data['Federation_Tier'], prefix='Fed').astype(int)
        test_federation_dummies = pd.get_dummies(self.test_data['Federation_Tier'], prefix='Fed').astype(int)

        # Drop one category to avoid multicollinearity
        train_federation_dummies = train_federation_dummies.drop('Fed_LOCAL', axis=1)
        test_federation_dummies = test_federation_dummies.drop('Fed_LOCAL', axis=1)

        base_features = [
            'Sex_encoded',
            'Equipment_Advantage',
            'BodyweightKg',
            'Tested',
            'Best3SquatKg',
            'Age',

        ]


        self.X_train = pd.concat([
            self.train_data[base_features],
            train_federation_dummies
        ], axis=1)



        self.X_test = pd.concat([
            self.test_data[base_features],
            test_federation_dummies
        ], axis=1)

        self.y_train = self.train_data['Strength']
        self.y_test =  self.test_data['Strength']
        return  self.X_train , self.X_test,self.y_train,self.y_test



    def prepare_features_classification(self):
        """Prepare features for modeling"""
        le = LabelEncoder()
        self.train_data['Sex_encoded'] = le.fit_transform(self.train_data['Sex'])
        self.test_data['Sex_encoded'] = le.transform(self.test_data['Sex'])

        # Create dummies for federation tiers
        train_federation_dummies = pd.get_dummies(self.train_data['Federation_Tier'], prefix='Fed').astype(int)
        test_federation_dummies = pd.get_dummies(self.test_data['Federation_Tier'], prefix='Fed').astype(int)

        # Drop one category to avoid multicollinearity
        train_federation_dummies = train_federation_dummies.drop('Fed_LOCAL', axis=1)
        test_federation_dummies = test_federation_dummies.drop('Fed_LOCAL', axis=1)

        self.train_data['Strength']=self.train_data['TotalKg']/self.train_data['BodyweightKg']
        self.test_data['Strength']=self.test_data['TotalKg']/self.test_data['BodyweightKg']

        base_features = [
            'Sex_encoded',
            'Age',
            'Strength',
            'Year',
        ]


        self.X_train = pd.concat([
            self.train_data[base_features],
            train_federation_dummies
        ], axis=1)



        self.X_test = pd.concat([
            self.test_data[base_features],
            test_federation_dummies
        ], axis=1)

        self.y_train = self.train_data['Tested']
        self.y_test =  self.test_data['Tested']
        return  self.X_train , self.X_test,self.y_train,self.y_test











class DataAnalyzer:
    def __init__(self, features: pd.DataFrame):
        """
        Initialize DataAnalyzer with pre-encoded feature data.

        Args:
            features (pd.DataFrame): Pre-encoded feature DataFrame

        Raises:
            ValueError: If features DataFrame is empty
        """
        if features.empty:
            raise ValueError("Features DataFrame cannot be empty")

        self.features = features
        self.numeric_columns = self._get_numeric_columns()
        self.scaler = StandardScaler()

    def _get_numeric_columns(self) -> List[str]:
        """Get numeric columns from features."""
        return list(self.features.select_dtypes(include='number').columns)

    def _scale_data(self, X: pd.DataFrame) -> np.ndarray:
        """Scale the input data using StandardScaler."""
        return self.scaler.fit_transform(X)

    def random_forest_analysis(self, y: pd.Series) -> pd.DataFrame:
        """
        Perform Random Forest feature importance analysis.

        Args:
            y (pd.Series): Target variable

        Returns:
            pd.DataFrame: DataFrame containing feature importance scores
        """
        if len(self.numeric_columns) == 0:
            raise ValueError("No numeric features available for analysis")

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        X = self.features[self.numeric_columns]

        rf.fit(X, y)
        feature_importances = pd.DataFrame({
            'feature': self.numeric_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importances

    def pca_and_condition_number(self) -> Tuple[np.ndarray, float,float]:
        """
        Perform PCA and calculate condition number on scaled numeric columns.

        Returns:
            Tuple[np.ndarray, float]: (explained variance ratios, condition number)
        """
        X = self.features[self.numeric_columns]

        X_scaled = self._scale_data(X)
        initialcondition_number = np.linalg.cond(X_scaled)

        pca = PCA()
        pca.fit(X_scaled)
        condition_number = np.linalg.cond(X_scaled)

        return pca.explained_variance_ratio_,initialcondition_number ,condition_number



    def svd_analysis(self, numerical_cols: List[str], categorical_cols: List[str]) -> Tuple[
        np.ndarray, pd.DataFrame, Dict]:
        """
        Perform comprehensive SVD analysis with feature-component correlations and visualizations.

        Args:
            numerical_cols: List of numerical column names to be scaled
            categorical_cols: List of categorical column names (already encoded)

        Returns:
            Tuple containing:
            - singular_values: Array of singular values
            - component_correlations: DataFrame showing feature-component correlations
            - analysis_metrics: Dictionary containing relative and cumulative importance
        """
        # Combine features
        X_num = self.features[numerical_cols]
        X_num_scaled = self._scale_data(X_num)
        X_cat = self.features[categorical_cols]
        X_combined = np.concatenate([X_num_scaled, X_cat], axis=1)

        # Perform SVD
        U, singular_values, Vt = svd(X_combined, full_matrices=False)

        # Calculate relative and cumulative importance
        total_variance = (singular_values ** 2).sum()
        relative_importance = (singular_values ** 2) / total_variance
        cumulative_importance = np.cumsum(relative_importance)

        # Create DataFrame of feature-component correlations
        all_features = numerical_cols + categorical_cols
        component_correlations = pd.DataFrame(
            Vt.T,
            columns=[f'Component_{i + 1}' for i in range(len(singular_values))],
            index=all_features
        )

        # Visualization 1: Singular Values Distribution
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(singular_values) + 1), singular_values, 'bo-')
        plt.xlabel('Component')
        plt.ylabel('Singular Value')
        plt.title('Singular Values Distribution')
        plt.grid(True)
        plt.show()

        # Visualization 2: Component Importance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(relative_importance) + 1),
                 relative_importance, 'bo-', label='Individual')
        plt.plot(range(1, len(cumulative_importance) + 1),
                 cumulative_importance, 'ro-', label='Cumulative')
        plt.xlabel('Component')
        plt.ylabel('Relative Importance')
        plt.title('SVD Component Importance')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Visualization 3: Feature-Component Correlations Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(component_correlations, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature-Component Correlations')
        plt.tight_layout()
        plt.show()

        # Create analysis metrics dictionary
        analysis_metrics = {
            'relative_importance': relative_importance,
            'cumulative_importance': cumulative_importance,
            'feature_importance_by_component': {}
        }

        # Add feature importance rankings for each component
        for comp in component_correlations.columns:
            analysis_metrics['feature_importance_by_component'][comp] = {
                'top_features': component_correlations[comp].abs().sort_values(ascending=False).head(),
                'correlations': component_correlations[comp].sort_values(ascending=False).head()
            }

        return singular_values, component_correlations, analysis_metrics

    def vif_analysis(self, numerical_cols: List[str]) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factors (VIF) for specified numeric columns.

        Args:
            numerical_cols: List of numerical column names to analyze

        Returns:
            pd.DataFrame: DataFrame containing VIF scores for each feature

        Raises:
            ValueError: If perfect multicollinearity is detected or columns not found
        """
        # Validate columns exist in the dataset
        if not all(col in self.features.columns for col in numerical_cols):
            raise ValueError("Some specified columns not found in the dataset")

        # Get numerical features
        X = self.features[numerical_cols]
        vifs = pd.DataFrame()
        vifs["feature"] = numerical_cols

        try:
            vifs["VIF"] = [variance_inflation_factor(X.values, i)
                           for i in range(X.shape[1])]
        except np.linalg.LinAlgError:
            raise ValueError("Perfect multicollinearity detected in features")

        return vifs.sort_values('VIF', ascending=False)








def create_svd_visualizations(singular_values):
    # Calculate relative and cumulative importance
    total_variance = (singular_values ** 2).sum()
    relative_importance = (singular_values ** 2) / total_variance
    cumulative_importance = np.cumsum(relative_importance)

    # 1. Singular Values Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(singular_values) + 1), singular_values, 'bo-')
    plt.xlabel('Component')
    plt.ylabel('Singular Value')
    plt.title('Singular Values Distribution')
    plt.grid(True)
    plt.show()

    # 2. Relative and Cumulative Importance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(relative_importance) + 1),
             relative_importance, 'bo-', label='Individual')
    plt.plot(range(1, len(cumulative_importance) + 1),
             cumulative_importance, 'ro-', label='Cumulative')
    plt.xlabel('Component')
    plt.ylabel('Relative Importance')
    plt.title('SVD Component Importance')
    plt.legend()
    plt.grid(True)
    plt.show()

    return relative_importance, cumulative_importance


def create_pca_visualizations(explained_variance_ratio, condition_number):
    # Set figure style parameters
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['axes.grid'] = True

    # 1. Scree Plot with Cumulative Variance
    fig, ax = plt.subplots()

    # Bar plot of individual explained variance
    bars = ax.bar(range(1, len(explained_variance_ratio) + 1),
                  explained_variance_ratio,
                  alpha=0.6,
                  color='skyblue',
                  label='Individual explained variance')

    # Line plot of cumulative explained variance
    line = ax.plot(range(1, len(explained_variance_ratio) + 1),
                   np.cumsum(explained_variance_ratio),
                   'ro-',
                   label='Cumulative explained variance')

    # Add percentage labels on top of bars
    for rect in bars:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height,
                f'{height:.1%}',
                ha='center', va='bottom')

    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Scree Plot with Cumulative Explained Variance')
    ax.legend()

    plt.tight_layout()
    plt.show()

    # 2. Pie Chart of Variance Distribution
    fig, ax = plt.subplots()
    colors = ['lightblue', 'lightgreen', 'lightpink', 'wheat']

    patches, texts, autotexts = ax.pie(explained_variance_ratio,
                                       labels=[f'PC{i + 1}\n({var:.1%})' for i, var in
                                               enumerate(explained_variance_ratio)],
                                       autopct='%1.1f%%',
                                       colors=colors)

    ax.set_title('Distribution of Variance Across Principal Components')
    plt.show()

    # 3. Horizontal Bar Chart comparing components
    fig, ax = plt.subplots()
    components = [f'PC{i + 1}' for i in range(len(explained_variance_ratio))]
    bars = ax.barh(components, explained_variance_ratio, color='lightblue')

    # Add percentage labels on bars
    for rect in bars:
        width = rect.get_width()
        ax.text(width, rect.get_y() + rect.get_height() / 2.,
                f'{width:.1%}',
                ha='left', va='center')

    ax.set_xlabel('Explained Variance Ratio')
    ax.set_title('Variance Explained by Each Principal Component')

    plt.tight_layout()
    plt.show()





class SVDTransformer:
    def __init__(self, n_components: int, numerical_cols: List[str], categorical_cols: List[str]):
        self.n_components = n_components
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler()
        self.Vt = None
        self.singular_values = None
        self.mean = None

    def fit(self, X: pd.DataFrame) -> 'SVDTransformer':
        if not all(col in X.columns for col in self.numerical_cols + self.categorical_cols):
            raise ValueError("Some specified columns not found in the dataset")

        X_num = X[self.numerical_cols]
        X_num_scaled = self.scaler.fit_transform(X_num)

        X_cat = X[self.categorical_cols]

        X_combined = np.concatenate([X_num_scaled, X_cat], axis=1)

        _, self.singular_values, self.Vt = svd(X_combined, full_matrices=False)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.Vt is None:
            raise ValueError("Transformer must be fitted before transform")

        X_num = X[self.numerical_cols]
        X_num_scaled = self.scaler.transform(X_num)

        X_cat = X[self.categorical_cols]

        X_combined = np.concatenate([X_num_scaled, X_cat], axis=1)

        return np.dot(X_combined, self.Vt[:self.n_components].T)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)



class StepwiseRegresser:
    def __init__(self, train_data: np.ndarray, test_data: np.ndarray,
                 y_train: pd.Series, y_test: pd.Series,
                 totalKg_train: pd.Series, totalKg_test: pd.Series,
                 bodyweight_Kg_train: pd.Series, bodyweight_Kg_test: pd.Series):

        self.X_train = train_data
        self.X_test = test_data
        self.y_train_unscaled = y_train.values
        self.y_test_unscaled = y_test.values
        self.TotalKg_Train = totalKg_train.values
        self.TotalKg_Test = totalKg_test.values
        self.BodywtKg_Train = bodyweight_Kg_train.values
        self.BodywtKg_Test = bodyweight_Kg_test.values

        self.y_scaler = StandardScaler()
        self.y_train = self.y_scaler.fit_transform(self.y_train_unscaled.reshape(-1, 1)).ravel()
        self.y_test = self.y_scaler.transform(self.y_test_unscaled.reshape(-1, 1)).ravel()


    def find_best_polynomial_degree(self, max_degree: int = 3) -> int:
        best_r2 = -np.inf
        best_degree = 0

        for degree in range(max_degree + 1):
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(self.X_train)
            X_test_poly = poly.transform(self.X_test)

            model = sm.OLS(self.y_train, X_train_poly).fit()
            y_pred = model.predict(X_test_poly)
            r2 = r2_score(self.y_test, y_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_degree = degree

        self.best_degree = best_degree
        return best_degree

    def perform_backward_stepwise(self, p_threshold: float = 0.05):

        poly = PolynomialFeatures(degree=self.best_degree)
        X_train_poly = poly.fit_transform(self.X_train)
        X_test_poly = poly.transform(self.X_test)


        # Start with all features
        included = list(range(X_train_poly.shape[1]))

        while True:
            if len(included) == 0:
                break

            X_subset = X_train_poly[:, included]
            model = sm.OLS(self.y_train, X_subset).fit()


            # Find maximum p-value
            max_pval = model.pvalues.max()

            # If max p-value > threshold, remove that feature
            if max_pval > p_threshold:
                print("Removing statistically insignificant feature")
                max_pval_idx = model.pvalues.argmax()
                included.pop(max_pval_idx)
            else:
                print("All coefficients are statistically significant")
                break

        # Fit final model
        X_subset_train = X_train_poly[:, included]
        X_subset_test = X_test_poly[:, included]
        final_model = sm.OLS(self.y_train, X_subset_train).fit()

        # Get predictions
        y_pred_train = final_model.predict(X_subset_train)
        y_pred_test = final_model.predict(X_subset_test)


        y_pred_train_orig = self.y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
        y_pred_test_orig = self.y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

        # Calculate TotalKg predictions
        totalkg_pred_train = y_pred_train_orig * self.BodywtKg_Train
        totalkg_pred_test = y_pred_test_orig * self.BodywtKg_Test



        # Print Strength metrics
        print("\nStrength Metrics:")
        print(f"R² Train: {r2_score(self.y_train, y_pred_train):.4f}")
        print(f"R² Test: {r2_score(self.y_test, y_pred_test):.4f}")
        print(f"MSE Train: {mean_squared_error(self.y_train, y_pred_train):.4f}")
        print(f"MSE Test: {mean_squared_error(self.y_test, y_pred_test):.4f}")
        print(f"RMSE Train: {np.sqrt(mean_squared_error(self.y_train, y_pred_train)):.4f}")
        print(f"RMSE Test: {np.sqrt(mean_squared_error(self.y_test, y_pred_test)):.4f}")
        print(f"AIC: {final_model.aic:.4f}")
        print(f"BIC: {final_model.bic:.4f}")
        print(f"Adjusted R²: {final_model.rsquared_adj:.4f}")
        print(f"F-test p-value: {final_model.f_pvalue:.4f}")



        # Print TotalKg metrics
        print("\nTotalKg Metrics:")
        print(f"R² Train: {r2_score(self.TotalKg_Train, totalkg_pred_train):.4f}")
        print(f"R² Test: {r2_score(self.TotalKg_Test, totalkg_pred_test):.4f}")
        print(f"MSE Train: {mean_squared_error(self.TotalKg_Train, totalkg_pred_train):.4f}")
        print(f"MSE Test: {mean_squared_error(self.TotalKg_Test, totalkg_pred_test):.4f}")
        print(f"RMSE Train: {np.sqrt(mean_squared_error(self.TotalKg_Train, totalkg_pred_train)):.4f}")
        print(f"RMSE Test: {np.sqrt(mean_squared_error(self.TotalKg_Test, totalkg_pred_test)):.4f}")


        #print with confident interval
        predictions = final_model.get_prediction(X_subset_test)
        pred_mean = predictions.predicted_mean
        ci = predictions.conf_int(alpha=0.05)

        # Create a plot
        plt.figure(figsize=(10, 6))

        # Plot actual values
        plt.scatter(self.y_test, self.y_test, color='black', label='Actual', alpha=0.5)

        # Plot predictions with confidence intervals
        plt.plot(self.y_test, pred_mean, color='blue', label='Predicted')
        plt.fill_between(self.y_test,
                         ci[:, 0],
                         ci[:, 1],
                         color='blue',
                         alpha=0.1,
                         label='95% CI')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Model Predictions with 95% Confidence Intervals')
        plt.legend()

        # Add diagonal line for perfect predictions
        plt.plot([min(self.y_test), max(self.y_test)],
                 [min(self.y_test), max(self.y_test)],
                 'k--', alpha=0.5)

        plt.show()

        # Print model summary
        print("\nModel Summary:")
        print(final_model.summary())

        """Plot actual vs predicted values for both train and test sets"""
        plt.figure(figsize=(15, 6))

        # Plot for Strength
        plt.subplot(1, 2, 1)
        # Training data
        plt.scatter(self.y_train_unscaled, y_pred_train_orig,
                    alpha=0.3, label='Train', color='blue')
        # Test data
        plt.scatter(self.y_test_unscaled, y_pred_test_orig,
                    alpha=0.3, label='Test', color='red')

        # Perfect prediction line
        min_val = min(min(self.y_train_unscaled), min(self.y_test_unscaled))
        max_val = max(max(self.y_train_unscaled), max(self.y_test_unscaled))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

        plt.xlabel('Actual Strength')
        plt.ylabel('Predicted Strength')
        plt.title('Strength: Actual vs Predicted')
        plt.legend()

        # Plot for TotalKg
        plt.subplot(1, 2, 2)
        # Training data
        plt.scatter(self.TotalKg_Train, totalkg_pred_train,
                    alpha=0.3, label='Train', color='blue')
        # Test data
        plt.scatter(self.TotalKg_Test, totalkg_pred_test,
                    alpha=0.3, label='Test', color='red')

        # Perfect prediction line
        min_val = min(min(self.TotalKg_Train), min(self.TotalKg_Test))
        max_val = max(max(self.TotalKg_Train), max(self.TotalKg_Test))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

        plt.xlabel('Actual TotalKg')
        plt.ylabel('Predicted TotalKg')
        plt.title('TotalKg: Actual vs Predicted')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return final_model



    def perform_xgb_regression(self):
        # Create polynomial features
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(self.X_train)
        X_test_poly = poly.transform(self.X_test)

        # Define XGBoost parameters
        params = {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.1,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 1.0,
            'gamma': 0.1
        }

        # Create and fit XGBoost model
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_train_poly, self.y_train)

        # Get predictions
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)

        # Transform predictions back to original scale
        y_pred_train_orig = self.y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
        y_pred_test_orig = self.y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

        # Calculate TotalKg predictions
        totalkg_pred_train = y_pred_train_orig * self.BodywtKg_Train
        totalkg_pred_test = y_pred_test_orig * self.BodywtKg_Test

        # Print Strength metrics
        print("\nStrength Metrics:")
        print(f"R² Train: {r2_score(self.y_train, y_pred_train):.4f}")
        print(f"R² Test: {r2_score(self.y_test, y_pred_test):.4f}")
        print(f"MSE Train: {mean_squared_error(self.y_train, y_pred_train):.4f}")
        print(f"MSE Test: {mean_squared_error(self.y_test, y_pred_test):.4f}")
        print(f"RMSE Train: {np.sqrt(mean_squared_error(self.y_train, y_pred_train)):.4f}")
        print(f"RMSE Test: {np.sqrt(mean_squared_error(self.y_test, y_pred_test)):.4f}")

        # Print TotalKg metrics
        print("\nTotalKg Metrics:")
        print(f"R² Train: {r2_score(self.TotalKg_Train, totalkg_pred_train):.4f}")
        print(f"R² Test: {r2_score(self.TotalKg_Test, totalkg_pred_test):.4f}")
        print(f"MSE Train: {mean_squared_error(self.TotalKg_Train, totalkg_pred_train):.4f}")
        print(f"MSE Test: {mean_squared_error(self.TotalKg_Test, totalkg_pred_test):.4f}")
        print(f"RMSE Train: {np.sqrt(mean_squared_error(self.TotalKg_Train, totalkg_pred_train)):.4f}")
        print(f"RMSE Test: {np.sqrt(mean_squared_error(self.TotalKg_Test, totalkg_pred_test)):.4f}")

        # Plot predictions
        plt.figure(figsize=(15, 6))

        # Plot for Strength
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_train_unscaled, y_pred_train_orig,
                    alpha=0.3, label='Train', color='blue')
        plt.scatter(self.y_test_unscaled, y_pred_test_orig,
                    alpha=0.3, label='Test', color='red')

        min_val = min(min(self.y_train_unscaled), min(self.y_test_unscaled))
        max_val = max(max(self.y_train_unscaled), max(self.y_test_unscaled))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

        plt.xlabel('Actual Strength')
        plt.ylabel('Predicted Strength')
        plt.title('XGBoost - Strength: Actual vs Predicted')
        plt.legend()

        # Plot for TotalKg
        plt.subplot(1, 2, 2)
        plt.scatter(self.TotalKg_Train, totalkg_pred_train,
                    alpha=0.3, label='Train', color='blue')
        plt.scatter(self.TotalKg_Test, totalkg_pred_test,
                    alpha=0.3, label='Test', color='red')

        min_val = min(min(self.TotalKg_Train), min(self.TotalKg_Test))
        max_val = max(max(self.TotalKg_Train), max(self.TotalKg_Test))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

        plt.xlabel('Actual TotalKg')
        plt.ylabel('Predicted TotalKg')
        plt.title('XGBoost - TotalKg: Actual vs Predicted')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return model


def plot_correlation_covariance_matrices(X_transformed: np.ndarray, feature_names: list = None) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Generate and display heatmaps for covariance and correlation matrices for transformed data.
    Args:
        X_transformed: Transformed data array
        feature_names: Optional list of feature names to use as labels
    """


    # If feature names not provided, create generic names
    if feature_names is None:
        feature_names = [f'Feature {i + 1}' for i in range(X_transformed.shape[1])]

    # Direct computation for transformed data
    n_samples = X_transformed.shape[0]
    covariance_matrix = (X_transformed.T @ X_transformed) / (n_samples - 1)

    # Convert to DataFrame with feature names
    covariance_df = pd.DataFrame(
        covariance_matrix,
        index=feature_names,
        columns=feature_names
    )

    # For correlation, normalize covariance by standard deviations
    std = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std, std)

    correlation_df = pd.DataFrame(
        correlation_matrix,
        index=feature_names,
        columns=feature_names
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(covariance_df,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                ax=ax1)
    ax1.set_title('Sample Covariance Matrix')

    sns.heatmap(correlation_df,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                vmin=-1, vmax=1,
                ax=ax2)
    ax2.set_title('Sample Correlation Matrix')

    plt.tight_layout()
    plt.show()



    return covariance_matrix, correlation_matrix





class MultiClassifierOptimizer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = self.X_train.columns
        self.results = {}
        self.models = {}

    def preprocess_data(self):
        for column in self.X_train.select_dtypes(['object']).columns:
            le = LabelEncoder()
            self.X_train[column] = le.fit_transform(self.X_train[column])
            self.X_test[column] = le.transform(self.X_test[column])


        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def apply_smote(self):
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def visualize_classification_results(self, y_true, y_pred, model_name):
        plt.figure(figsize=(20, 6))

        plt.subplot(131)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Tested', 'Tested'],
                    yticklabels=['Not Tested', 'Tested'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.subplot(132)
        df_results = pd.DataFrame({
            'True': y_true,
            'Predicted': y_pred
        })
        correct = (y_true == y_pred)

        classification_results = pd.DataFrame({
            'Category': ['Correct Not Tested', 'Correct Tested',
                         'Misclassified Not Tested', 'Misclassified Tested'],
            'Count': [
                sum((correct) & (y_true == 0)),
                sum((correct) & (y_true == 1)),
                sum((~correct) & (y_true == 0)),
                sum((~correct) & (y_true == 1))
            ]
        })

        sns.barplot(data=classification_results, x='Category', y='Count')
        plt.xticks(rotation=45)
        plt.title('Classification Distribution')

        plt.subplot(133)
        error_types = {
            'False Positives': cm[0, 1],
            'False Negatives': cm[1, 0],
            'True Positives': cm[1, 1],
            'True Negatives': cm[0, 0]
        }

        colors = ['red', 'orange', 'green', 'blue']
        plt.pie(error_types.values(), labels=error_types.keys(),
                autopct='%1.1f%%', colors=colors)
        plt.title('Classification Analysis')

        plt.tight_layout()
        plt.show()

        print(f"\nDetailed Classification Analysis for {model_name}:")
        print(f"Total Samples: {len(y_true)}")
        print(f"Correctly Classified: {sum(correct)} ({sum(correct) / len(y_true) * 100:.2f}%)")
        print(f"Misclassified: {sum(~correct)} ({sum(~correct) / len(y_true) * 100:.2f}%)")
        print("\nPer-Class Analysis:")
        for class_name, true_val in zip(['Not Tested', 'Tested'], [0, 1]):
            class_total = sum(y_true == true_val)
            class_correct = sum((y_true == true_val) & correct)
            print(f"\n{class_name}:")
            print(f"Total: {class_total}")
            print(f"Correctly Classified: {class_correct} ({class_correct / class_total * 100:.2f}%)")
            print(
                f"Misclassified: {class_total - class_correct} ({(class_total - class_correct) / class_total * 100:.2f}%)")

    def evaluate_classifier(self, model, model_name, X_train=None, y_train=None):
        X_train = X_train if X_train is not None else self.X_train
        y_train = y_train if y_train is not None else self.y_train

        # Stratified K-fold Cross Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf)
        print(f"\n{model_name} Cross-Validation Scores:")
        table_data = []
        for i, score in enumerate(cv_scores, start=1):
            table_data.append([f"CV {i}", score])

        # Display the table
        print("Stratified CV Scores:")
        print(tabulate(table_data, headers=["Fold", "Score"], tablefmt="grid"))
        print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Get predictions
        y_pred = model.predict(self.X_test)

        # Visualize classification results
        self.visualize_classification_results(self.y_test, y_pred, model_name)

        # Calculate metrics
        cm = confusion_matrix(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        f1 = f1_score(self.y_test, y_pred)

        print(f"\nClassification Metrics for {model_name}:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall (Sensitivity): {recall:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print(f"F1-Score: {f1:.3f}")

        # ROC Curve and AUC
        try:
            y_prob = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc_score = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.show()

            roc_data = {'fpr': fpr, 'tpr': tpr}
        except:
            print("ROC/AUC not available for this model")
            auc_score = None
            roc_data = None

        # Per-class metrics visualization
        plt.figure(figsize=(10, 6))
        class_names = ['Not Tested', 'Tested']
        metrics_data = {
            'Class': class_names,
            'Precision': [precision_score(self.y_test, y_pred, pos_label=i) for i in [0, 1]],
            'Recall': [recall_score(self.y_test, y_pred, pos_label=i) for i in [0, 1]],
            'F1-score': [f1_score(self.y_test, y_pred, pos_label=i) for i in [0, 1]]
        }
        metrics_df = pd.DataFrame(metrics_data)

        metrics_melted = pd.melt(metrics_df, id_vars=['Class'], var_name='Metric', value_name='Score')
        sns.barplot(data=metrics_melted, x='Class', y='Score', hue='Metric')
        plt.title(f'Classification Metrics by Class - {model_name}')
        plt.show()

        return {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'cv_scores': cv_scores,
            'auc_score': auc_score,
            'roc_curve': roc_data  # Added ROC curve data
        }
    def optimize_decision_tree(self):
        # Pre-pruning phase
        print("\n=== Pre-pruning Phase ===")
        '''
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None],
            'min_samples_leaf': [1, 2, 4],
            'min_impurity_decrease': [0.0, 0.01]
        }'''
        # best parameters obtained :-
        # {'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 2, 'splitter': 'best'}
        # since I have already obtained best parameters have from running the param above provied those values in grid to save time, for cross checking you can commenet out this
        #line and use grid  with hyperparamers in the above param grid which is commented out  above
        param_grid = {
            'criterion': ['gini'],
            'splitter': ['best'],
            'max_depth': [10],
            'min_samples_split': [2],
            'max_features':  [None],
            'min_samples_leaf':  [2],
            'min_impurity_decrease': [0.0],
        }
      #best parameters obtained :-
        #{'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 2, 'splitter': 'best'}

        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(dt, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_train, self.y_train)

        # Evaluate pre-pruned model
        pre_pruned_dt = DecisionTreeClassifier(random_state=42, **grid_search.best_params_)
        pre_pruned_dt.fit(self.X_train, self.y_train)
        print("\nPre-pruning Best Parameters:")
        print(grid_search.best_params_)

        pre_pruning_results = self.evaluate_classifier(
            pre_pruned_dt,
            'Decision Tree (Pre-pruning)',
            self.X_train,
            self.y_train
        )


        # Post-pruning phase
        print("\n=== Post-pruning Phase ===")
        path = pre_pruned_dt.cost_complexity_pruning_path(self.X_train, self.y_train)
        ccp_alphas = path.ccp_alphas[:-1:10]  # Remove the last value
        print(len(ccp_alphas))
        # Train trees with different ccp values
        dt_scores = []
        for ccp_alpha in ccp_alphas:
            dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha, **grid_search.best_params_)
            scores = cross_val_score(dt, self.X_train, self.y_train, cv=5)
            print("performing for alpha " + str(ccp_alpha) + " score " + str(scores))
            dt_scores.append(np.mean(scores))

        # Plot cost complexity pruning path
        plt.figure(figsize=(10, 6))
        plt.plot(ccp_alphas, dt_scores, marker='o')
        plt.xlabel('Cost Complexity (alpha)')
        plt.ylabel('Mean Cross-validated Accuracy')
        plt.title('Cost Complexity Pruning Path')
        plt.show()

        # Find optimal ccp_alpha
        optimal_ccp_alpha = ccp_alphas[np.argmax(dt_scores)]
        print(f"\nOptimal cost complexity parameter: {optimal_ccp_alpha}")

        # Final model with both pre and post pruning
        final_params = grid_search.best_params_
        final_params['ccp_alpha'] = optimal_ccp_alpha

        self.models['DecisionTree'] = DecisionTreeClassifier(random_state=42, **final_params)
        self.models['DecisionTree'].fit(self.X_train, self.y_train)

        post_pruning_results = self.evaluate_classifier(
            self.models['DecisionTree'],
            'Decision Tree (Post-pruning)',
            self.X_train,
            self.y_train
        )

        # Compare pre and post pruning results
        print("\n=== Pruning Comparison ===")
        comparison_metrics = ['precision', 'recall', 'specificity', 'f1_score', 'auc_score']
        pruning_comparison = pd.DataFrame({
            'Pre-pruning': [pre_pruning_results[metric] for metric in comparison_metrics],
            'Post-pruning': [post_pruning_results[metric] for metric in comparison_metrics]
        }, index=comparison_metrics)

        print("\nPre vs Post Pruning Performance:")
        print(pruning_comparison.round(3))

        # Store final results
        self.results['DecisionTree'] = post_pruning_results

    def optimize_logistic_regression(self):
        lr = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
        lr.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = lr
        evaluation_results = self.evaluate_classifier(
            lr,
            'Logistic Regression',
            self.X_train,
            self.y_train
        )
        self.results['logistic_regression'] = evaluation_results

    def optimize_knn(self):
        k_range = range(1, 15, 2)
        k_scores = []

        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            scores = cross_val_score(knn, self.X_train, self.y_train, cv=5)
            k_scores.append(scores.mean())

        plt.plot(k_range, k_scores)
        plt.xlabel('K')
        plt.ylabel('Cross-validated accuracy')
        plt.title('Elbow Method for Optimal K')
        plt.show()

        optimal_k = k_range[np.argmax(k_scores)]
        self.models['knn'] = KNeighborsClassifier(n_neighbors=optimal_k).fit(self.X_train, self.y_train)
        evaluation_results = self.evaluate_classifier(
            self.models['knn'],
            'K-Nearest Neighbors',
            self.X_train,
            self.y_train
        )
        self.results['knn'] = evaluation_results

    def optimize_svm(self):
        from sklearn.calibration import CalibratedClassifierCV

        sample_size = 10000
        unique, counts = np.unique(self.y_train, return_counts=True)
        class_props = counts / len(self.y_train)

        stratified_indices = []
        for class_val, prop in zip(unique, class_props):
            class_indices = np.where(self.y_train == class_val)[0]
            n_samples = int(sample_size * prop)
            sampled_indices = np.random.choice(class_indices, n_samples, replace=False)
            stratified_indices.extend(sampled_indices)

        X_sample = self.X_train[stratified_indices]
        y_sample = self.y_train[stratified_indices]

        print("\nClass Distribution:")
        print("Original:", dict(zip(unique, counts / len(self.y_train))))
        print("Sampled:", dict(zip(*np.unique(y_sample, return_counts=True))))

        # Create base LinearSVC
        base_linear_svc = LinearSVC(
            random_state=42,
            max_iter=2000,
            dual=False,
            C=1.0
        )

        # Wrap with CalibratedClassifierCV to get probability estimates
        linear_svc = CalibratedClassifierCV(base_linear_svc, cv=3)

        skf = StratifiedKFold(n_splits=3)
        linear_scores = cross_val_score(linear_svc, X_sample, y_sample, cv=skf, n_jobs=-1)
        print(f"\nLinear SVM CV Scores: {linear_scores.mean():.3f} (+/- {linear_scores.std() * 2:.3f})")

        linear_svc.fit(X_sample, y_sample)
        self.models['svm_linear'] = linear_svc
        evaluation_results_linear = self.evaluate_classifier(
            linear_svc,
            'SVM Linear',
            X_sample,
            y_sample
        )
        self.results['svm_linear'] = evaluation_results_linear

        rbf_svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            max_iter=2000,
            cache_size=2000,
            probability=True
        )
        rbf_scores = cross_val_score(rbf_svm, X_sample, y_sample, cv=skf, n_jobs=-1)
        print(f"\nRBF SVM CV Scores: {rbf_scores.mean():.3f} (+/- {rbf_scores.std() * 2:.3f})")

        rbf_svm.fit(X_sample, y_sample)
        self.models['svm_rbf'] = rbf_svm
        evaluation_results_rbf = self.evaluate_classifier(
            rbf_svm,
            'SVM RBF',
            X_sample,
            y_sample
        )
        self.results['svm_rbf'] = evaluation_results_rbf

        # Polynomial SVM
        poly_svm = SVC(
            kernel='poly',
            degree=2,  # Polynomial degree
            C=1.0,
            gamma='scale',
            coef0=1,  # Independent term in polynomial function
            random_state=42,
            max_iter=2000,
            cache_size=2000,
            probability=True
        )
        poly_scores = cross_val_score(poly_svm, X_sample, y_sample, cv=skf, n_jobs=-1)
        print(f"\nPolynomial SVM CV Scores: {poly_scores.mean():.3f} (+/- {poly_scores.std() * 2:.3f})")

        poly_svm.fit(X_sample, y_sample)
        self.models['svm_poly'] = poly_svm
        evaluation_results_poly = self.evaluate_classifier(
            poly_svm,
            'SVM Polynomial',
            X_sample,
            y_sample
        )
        self.results['svm_poly'] = evaluation_results_poly

    def optimize_naive_bayes(self):
        nb = GaussianNB()
        nb.fit(self.X_train, self.y_train)
        self.models['naive_bayes'] = nb
        evaluation_results = self.evaluate_classifier(
            nb,
            'Naive Bayes',
            self.X_train,
            self.y_train
        )
        self.results['naive_bayes'] = evaluation_results

    def optimize_neural_network(self):
        sample_size = 10000

        unique, counts = np.unique(self.y_train, return_counts=True)
        class_props = counts / len(self.y_train)

        stratified_indices = []
        for class_val, prop in zip(unique, class_props):
            class_indices = np.where(self.y_train == class_val)[0]
            n_samples = int(sample_size * prop)
            sampled_indices = np.random.choice(class_indices, n_samples, replace=False)
            stratified_indices.extend(sampled_indices)

        X_sample = self.X_train[stratified_indices]
        y_sample = self.y_train[stratified_indices]

        print("\nNeural Network Class Distribution:")
        print("Original:", dict(zip(unique, counts / len(self.y_train))))
        print("Sampled:", dict(zip(*np.unique(y_sample, return_counts=True))))

        mlp = MLPClassifier(
            hidden_layer_sizes=(50,),
            activation='relu',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=5,
            random_state=42
        )

        skf = StratifiedKFold(n_splits=3)
        scores = cross_val_score(mlp, X_sample, y_sample, cv=skf, n_jobs=-1)
        print(f"\nNeural Network CV Scores: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

        mlp.fit(X_sample, y_sample)
        self.models['neural_network'] = mlp
        evaluation_results = self.evaluate_classifier(
            mlp,
            'Neural Network',
            X_sample,
            y_sample
        )
        self.results['neural_network'] = evaluation_results



    def plot_results(self):
        # Create PrettyTable instance
        table = PrettyTable()

        # Add column headers
        table.field_names = ["Model", "Precision", "Recall", "Specificity", "F1-Score", "AUC"]

        # Set float precision
        float_format = ".3f"

        # Add rows for each model
        for model in self.results:
            table.add_row([
                model,
                f"{self.results[model]['precision']:{float_format}}",
                f"{self.results[model]['recall']:{float_format}}",
                f"{self.results[model]['specificity']:{float_format}}",
                f"{self.results[model]['f1_score']:{float_format}}",
                f"{self.results[model]['auc_score']:{float_format}}"
            ])

        # Set alignment
        table.align["Model"] = "l"  # left align model names
        table.align["Precision"] = "r"
        table.align["Recall"] = "r"
        table.align["Specificity"] = "r"
        table.align["F1-Score"] = "r"
        table.align["AUC"] = "r"

        print("\nModel Comparison Results:")
        print(table)

        # Plot ROC curves only for models with ROC data
        plt.figure(figsize=(10, 8))
        for model_name in self.results:
            if (self.results[model_name]['auc_score'] is not None and
                    'roc_curve' in self.results[model_name]):  # Check if ROC data exists
                fpr = self.results[model_name]['roc_curve']['fpr']
                tpr = self.results[model_name]['roc_curve']['tpr']
                plt.plot(
                    fpr, tpr,
                    label=f'{model_name} (AUC = {self.results[model_name]["auc_score"]:.2f})'
                )
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.show()

        metrics = ['precision', 'recall', 'specificity', 'f1_score', 'auc_score']
        comparison = pd.DataFrame([[self.results[model][metric] for metric in metrics] for model in self.results],
                                  columns=metrics, index=self.results.keys())


        best_model = comparison['auc_score'].idxmax()
        print(f"\nBest model based on AUC score: {best_model}")

        return comparison

    def optimize_all(self, use_smote=True):
        self.preprocess_data()
        plot_correlation_covariance_matrices(self.X_train,
                                             ['Sex_encoded', 'Age', 'Strength', 'Year', 'Fed_ELITE', 'Fed_MAJOR',
                                              'Fed_REGIONAL'])

        if use_smote:
            self.apply_smote()


        optimizations = [
            self.optimize_decision_tree,
            self.optimize_logistic_regression,
            self.optimize_knn,
            self.optimize_svm,
            self.optimize_naive_bayes,
            self.optimize_neural_network
        ]

        for optimization in optimizations:
            print(f"\nOptimizing {optimization.__name__}...")
            optimization()

        comparison = self.plot_results()
        return self.models, comparison


class ClusteringAndAssocialtionRuleMining:
    def __init__(self, data):
        """Initialize with powerlifting dataframe"""
        self.data = data
        self.prepare_features()

    def prepare_features(self):
        """Prepare strength ratio features"""
        self.data['SquatStren'] = self.data['Best3SquatKg'] / self.data['BodyweightKg']
        self.data['BenchStren'] = self.data['Best3BenchKg'] / self.data['BodyweightKg']
        self.data['DeadtStren'] = self.data['Best3DeadliftKg'] / self.data['BodyweightKg']

    def filter_data(self, sex='M', equipment='Raw', sample_size=None):
        """Filter and sample data"""
        filtered = self.data[self.data['Sex'] == sex]
        if equipment:
            filtered = filtered[filtered['Equipment'] == equipment]
        if sample_size:
            filtered = filtered.sample(n=min(sample_size, len(filtered)), random_state=42)
        return filtered

   #have reduced sample size to run fast
    def kmeans_analysis(self, sex='M', equipment=None, n_samples=10000, verbose=False):
        """Perform K-means clustering analysis"""
        filtered_data = self.filter_data(sex, equipment, n_samples)
        features = ['SquatStren', 'BenchStren', 'DeadtStren']
        X = filtered_data[features]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate metrics for different k
        max_clusters = 10
        range_n_clusters = range(1, max_clusters + 1)
        wss_scores = []
        silhouette_scores = []

        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_scaled)
            wss_scores.append(kmeans.inertia_)
            if n_clusters > 1:
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

        # Plot results
        self._plot_clustering_metrics(range_n_clusters, wss_scores, silhouette_scores)

        # Fit final model with k=8 (as per your analysis)
        final_kmeans = KMeans(n_clusters=8, random_state=42)
        clusters = final_kmeans.fit_predict(X_scaled)

        if verbose:
            self._print_cluster_stats(filtered_data, clusters, features)

        return final_kmeans, clusters

    def dbscan_analysis(self, sex='M', equipment='Raw', eps=0.3, min_samples=10, verbose=False):
        """Perform DBSCAN clustering analysis"""
        filtered_data = self.filter_data(sex, equipment)
        features = ['SquatStren', 'BenchStren', 'DeadtStren']
        X = filtered_data[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        clusters = dbscan.fit_predict(X_scaled)

        self._plot_dbscan_results(X_scaled, clusters)

        if verbose:
            self._print_cluster_stats(filtered_data, clusters, features)

        return dbscan, clusters

    def apriori_analysis(self, sex='M', equipment='Raw', n_samples=100000,
                         min_support=0.03, min_confidence=0.5):
        """Perform Apriori analysis"""
        filtered_data = self.filter_data(sex, equipment, n_samples)
        transactions = self._create_transactions(filtered_data)

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
            num_itemsets=len(frequent_itemsets)
        )

        self._print_rules_summary(rules)
        return rules

    def _create_transactions(self, data):
        """Create transactions for Apriori analysis"""
        transactions = []
        for _, row in data.iterrows():
            transaction = set()

            # Squat Categories
            if row['SquatStren'] <= 1.5:
                transaction.add('Squat_Novice')
            elif row['SquatStren'] <= 2.0:
                transaction.add('Squat_Intermediate')
            elif row['SquatStren'] <= 2.5:
                transaction.add('Squat_Advanced')
            else:
                transaction.add('Squat_Elite')

            # Bench Categories
            if row['BenchStren'] <= 1.0:
                transaction.add('Bench_Novice')
            elif row['BenchStren'] <= 1.5:
                transaction.add('Bench_Intermediate')
            elif row['BenchStren'] <= 2.0:
                transaction.add('Bench_Advanced')
            else:
                transaction.add('Bench_Elite')

            # Deadlift Categories
            if row['DeadtStren'] <= 1.8:
                transaction.add('Dead_Novice')
            elif row['DeadtStren'] <= 2.3:
                transaction.add('Dead_Intermediate')
            elif row['DeadtStren'] <= 2.8:
                transaction.add('Dead_Advanced')
            else:
                transaction.add('Dead_Elite')

            # Add other categories (age, weight, testing status)
            transactions.append(list(transaction))
        return transactions

    def _plot_clustering_metrics(self, range_n_clusters, wss_scores, silhouette_scores):
        """Plot K-means metrics"""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range_n_clusters, wss_scores, 'bo-')
        plt.title('Elbow Method using WSS')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(list(range_n_clusters)[1:], silhouette_scores, 'go-')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def _plot_dbscan_results(self, X_scaled, clusters):
        """Plot DBSCAN results"""
        plt.figure(figsize=(10, 6))
        plt.scatter(X_scaled[clusters != -1, 0],
                    X_scaled[clusters != -1, 1],
                    c=clusters[clusters != -1],
                    cmap='viridis',
                    label='Clustered points',
                    alpha=.05)
        plt.scatter(X_scaled[clusters == -1, 0],
                    X_scaled[clusters == -1, 1],
                    c='black',
                    marker='x',
                    label='Noise points')
        plt.xlabel('Standardized Squat Strength')
        plt.ylabel('Standardized Bench Strength')
        plt.title('DBSCAN Clustering of Powerlifting Data')
        plt.legend()
        plt.show()

    def _print_cluster_stats(self, data, clusters, features):
        """Print cluster statistics"""
        data['Cluster'] = clusters
        print("\nCluster Statistics:")
        for cluster in sorted(set(clusters)):
            if cluster == -1:
                print("\nNoise points:")
            else:
                print(f"\nCluster {cluster}:")
            cluster_data = data[data['Cluster'] == cluster]
            print(cluster_data[features].mean().round(2))
            print(f"Size: {len(cluster_data)}")

    def _print_rules_summary(self, rules):
        """Print association rules summary"""
        rules_sorted = rules.sort_values(['lift', 'confidence'], ascending=[False, False])

        print("\nTop Association Rules by Lift:")
        print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

        print("\nMost Common Associations (by Support):")
        print(rules_sorted.sort_values('support', ascending=False)[
                  ['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))






if __name__ == "__main__":
    processor = PowerliftingDataProcessor('openpowerlifting.csv')
    train_data, test_data = processor.process_data()
    EncoderClass = PrepareFeatures(train_data, test_data)




    en_train_data,en_test_data,y_train,y_test=EncoderClass.prepare_features_regression()


    #RandomForest
    data_analyzer1 = DataAnalyzer(en_train_data)
    rf_feature_importances = data_analyzer1.random_forest_analysis(y_train)
    print(rf_feature_importances)
    data_analyzer = DataAnalyzer(en_train_data[['Best3SquatKg','BodyweightKg','Age','Equipment_Advantage']])
    #PCA
    expVariance,initial_condition,condition_PCA= data_analyzer.pca_and_condition_number()
    #SVD
    singular_vals, correlations, metrics = data_analyzer1.svd_analysis(
        numerical_cols=['Best3SquatKg', 'BodyweightKg', 'Age'],
        categorical_cols=['Sex_encoded', 'Equipment_Advantage', 'Tested', 'Fed_REGIONAL', 'Fed_MAJOR', 'Fed_ELITE']
    )
    # Access the results:
    print("\nRelative Importance of Components:")
    print(metrics['relative_importance'])

    print("\nCumulative Importance:")
    print(metrics['cumulative_importance'])

    print("\nFeature Importance by Component:")
    for comp, data in metrics['feature_importance_by_component'].items():
        print(f"\n{comp} most important features (absolute correlation):")
        print(data['top_features'])
        print(f"\n{comp} correlations (with direction):")
        print(data['correlations'])

    create_svd_visualizations(singular_vals)

    print(expVariance)
    print(initial_condition)
    print(condition_PCA)
    create_pca_visualizations(expVariance,condition_PCA)


    print(data_analyzer1.vif_analysis(['Best3SquatKg', 'BodyweightKg', 'Age']))



    #APPLY SVD TRansformation
    svd_transformer = SVDTransformer(
        n_components=7,
        numerical_cols=['Best3SquatKg', 'BodyweightKg', 'Age'],
        categorical_cols=['Sex_encoded', 'Equipment_Advantage', 'Tested', 'Fed_REGIONAL', 'Fed_MAJOR', 'Fed_ELITE']

    )
    X_train_transformed = svd_transformer.fit_transform(en_train_data)
    X_test_transformed= svd_transformer.transform(en_test_data)
    plot_correlation_covariance_matrices(X_train_transformed)







    '''phase 2'''

    regresser= StepwiseRegresser(pd.DataFrame(X_train_transformed),pd.DataFrame(X_test_transformed),y_train,y_test,train_data['TotalKg'],test_data['TotalKg'],train_data['BodyweightKg'],test_data['BodyweightKg'])
    regresser.find_best_polynomial_degree(2)
    regresser.perform_backward_stepwise(.05)
    regresser.perform_xgb_regression()

    '''phase 2 end '''





    '''phase 3'''

    en_train_data,en_test_data,y_train,y_test=EncoderClass.prepare_features_classification()

    optimizer = MultiClassifierOptimizer(en_train_data, en_test_data, y_train, y_test)

    # Run complete optimization process
    best_model, feature_importance = optimizer.optimize_all(use_smote=True)

    '''phase 3'''



    '''phase 4'''
    analyzer = ClusteringAndAssocialtionRuleMining(train_data)

    # Run K-means
    kmeans_model, kmeans_clusters = analyzer.kmeans_analysis(verbose=True)

    # Run DBSCAN
    dbscan_model, dbscan_clusters = analyzer.dbscan_analysis(verbose=True)

    # Run Apriori
    rules = analyzer.apriori_analysis()



    '''phase 4 end '''



