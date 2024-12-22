# Step 1: Claims Frequency and Severity Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson, nbinom, geom, lognorm, gamma, ks_2samp, anderson

class ClaimAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def fit_poisson(self, column: str):
        """
        Fit a Poisson distribution to the specified column.

        Args:
            column (str): Column to analyze.

        Returns:
            dict: Estimated parameter and goodness-of-fit statistics.
        """
        mean_val = np.mean(self.data[column])
        lambda_hat = mean_val  # Poisson MLE for lambda is the mean

        # Goodness-of-fit using Chi-Square
        observed, bins = np.histogram(self.data[column], bins='auto', density=False)
        expected = poisson.pmf(np.arange(len(observed)), mu=lambda_hat) * len(self.data)
        chi_square = np.sum((observed - expected)**2 / expected)

        return {"lambda": lambda_hat, "chi_square": chi_square}

    def fit_negative_binomial(self, column: str):
        """
        Fit a Negative Binomial distribution to the specified column.

        Args:
            column (str): Column to analyze.

        Returns:
            dict: Estimated parameters and goodness-of-fit statistics.
        """
        mean_val = np.mean(self.data[column])
        var_val = np.var(self.data[column])

        # Estimation
        p_hat = mean_val / var_val
        r_hat = mean_val * p_hat / (1 - p_hat)

        # Goodness-of-fit using Chi-Square
        observed, bins = np.histogram(self.data[column], bins='auto', density=False)
        expected = nbinom.pmf(np.arange(len(observed)), n=r_hat, p=p_hat) * len(self.data)
        chi_square = np.sum((observed - expected)**2 / expected)

        return {"r": r_hat, "p": p_hat, "chi_square": chi_square}

    def fit_geometric(self, column: str):
        """
        Fit a Geometric distribution to the specified column.

        Args:
            column (str): Column to analyze.

        Returns:
            dict: Estimated parameters and goodness-of-fit statistics.
        """
        mean_val = np.mean(self.data[column])
        p_hat = 1 / (1 + mean_val)

        # Goodness-of-fit using Chi-Square
        observed, bins = np.histogram(self.data[column], bins='auto', density=False)
        expected = geom.pmf(np.arange(len(observed)), p=p_hat) * len(self.data)
        chi_square = np.sum((observed - expected)**2 / expected)

        return {"p": p_hat, "chi_square": chi_square}

    def fit_lognormal(self, column: str):
        """
        Fit a Lognormal distribution to the specified column.

        Args:
            column (str): Column to analyze.

        Returns:
            dict: Estimated parameters and goodness-of-fit statistics.
        """
        log_data = np.log(self.data[column][self.data[column] > 0])
        shape, loc, scale = lognorm.fit(log_data, floc=0)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = ks_2samp(self.data[column], lognorm.rvs(shape, loc=loc, scale=scale, size=len(self.data)))

        return {"shape": shape, "loc": loc, "scale": scale, "ks_stat": ks_stat, "ks_p_value": ks_p_value}

    def fit_gamma(self, column: str):
        """
        Fit a Gamma distribution to the specified column.

        Args:
            column (str): Column to analyze.

        Returns:
            dict: Estimated parameters and goodness-of-fit statistics.
        """
        shape, loc, scale = gamma.fit(self.data[column], floc=0)

        # Anderson-Darling test
        anderson_stat = anderson(self.data[column], dist='gamma')

        return {"shape": shape, "loc": loc, "scale": scale, "anderson_stat": anderson_stat}

# Usage example (to be replaced with actual data loading)
def main():
    # Replace with actual dataset
    sample_data = pd.DataFrame({
        "claim_frequency": np.random.poisson(2, 1000),
        "claim_severity": np.random.lognormal(mean=2, sigma=0.5, size=1000)
    })

    analysis = ClaimAnalysis(sample_data)

    # Fit distributions
    poisson_results = analysis.fit_poisson("claim_frequency")
    print("Poisson Results:", poisson_results)

    nb_results = analysis.fit_negative_binomial("claim_frequency")
    print("Negative Binomial Results:", nb_results)

    geometric_results = analysis.fit_geometric("claim_frequency")
    print("Geometric Results:", geometric_results)

    lognormal_results = analysis.fit_lognormal("claim_severity")
    print("Lognormal Results:", lognormal_results)

    gamma_results = analysis.fit_gamma("claim_severity")
    print("Gamma Results:", gamma_results)

if __name__ == "__main__":
    main()
# Step 2: Data Visualization and Advanced Statistics

class AdvancedAnalysis(ClaimAnalysis):
    def plot_histograms(self):
        """
        Plot histograms for all numerical columns in the dataset.
        """
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            plt.figure()
            sns.histplot(self.data[col], kde=True, color="purple")
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

    def calculate_correlation_matrix(self):
        """
        Compute and visualize the correlation matrix of numerical columns.
        """
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        correlation_matrix = self.data[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()
        return correlation_matrix

    def perform_anova(self, dependent_var: str, independent_var: str):
        """
        Perform one-way ANOVA test.

        Args:
            dependent_var (str): The dependent variable.
            independent_var (str): The independent variable.

        Returns:
            dict: ANOVA test statistics.
        """
        from scipy.stats import f_oneway
        groups = [group[dependent_var].values for name, group in self.data.groupby(independent_var)]
        f_stat, p_value = f_oneway(*groups)
        return {"F-statistic": f_stat, "p-value": p_value}

    def fit_pareto(self, column: str):
        """
        Fit a Pareto distribution to the specified column.

        Args:
            column (str): Column to analyze.

        Returns:
            dict: Estimated parameters and goodness-of-fit statistics.
        """
        from scipy.stats import pareto
        shape, loc, scale = pareto.fit(self.data[column], floc=0)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = ks_2samp(self.data[column], pareto.rvs(shape, loc=loc, scale=scale, size=len(self.data)))

        return {"shape": shape, "loc": loc, "scale": scale, "ks_stat": ks_stat, "ks_p_value": ks_p_value}

# Example extended functionality
def advanced_main():
    # Replace with actual dataset
    sample_data = pd.DataFrame({
        "claim_frequency": np.random.poisson(2, 1000),
        "claim_severity": np.random.lognormal(mean=2, sigma=0.5, size=1000),
        "region": np.random.choice(["North", "South", "East", "West"], 1000)
    })

    advanced_analysis = AdvancedAnalysis(sample_data)

    # Plot histograms
    advanced_analysis.plot_histograms()

    # Correlation matrix
    correlation_matrix = advanced_analysis.calculate_correlation_matrix()
    print("Correlation Matrix:\n", correlation_matrix)

    # ANOVA
    anova_results = advanced_analysis.perform_anova(dependent_var="claim_severity", independent_var="region")
    print("ANOVA Results:", anova_results)

    # Fit Pareto
    pareto_results = advanced_analysis.fit_pareto("claim_severity")
    print("Pareto Results:", pareto_results)

if __name__ == "__main__":
    advanced_main()

# Step 4: Enhancing Data Analysis with Reporting and Metrics

import statsmodels.api as sm

class ReportingAnalysis(AdvancedAnalysis):
    def generate_summary_statistics(self):
        """
        Generate and print summary statistics for numerical columns.
        """
        summary_stats = self.data.describe()
        print("Summary Statistics:\n", summary_stats)
        return summary_stats

    def linear_regression(self, dependent_var: str, independent_vars: list):
        """
        Perform a linear regression analysis.

        Args:
            dependent_var (str): The dependent variable.
            independent_vars (list): List of independent variables.

        Returns:
            Regression results summary.
        """
        X = self.data[independent_vars]
        X = sm.add_constant(X)  # Add intercept
        y = self.data[dependent_var]
        model = sm.OLS(y, X).fit()
        print(model.summary())
        return model.summary()

    def compute_aggregate_metrics(self, group_by_column: str):
        """
        Compute aggregate metrics like mean and sum grouped by a categorical column.

        Args:
            group_by_column (str): Column to group by.

        Returns:
            DataFrame: Grouped aggregate metrics.
        """
        grouped_metrics = self.data.groupby(group_by_column).agg(['mean', 'sum'])
        print("Aggregated Metrics:\n", grouped_metrics)
        return grouped_metrics

# Example usage for extended analysis
def reporting_main():
    # Replace with actual dataset
    sample_data = pd.DataFrame({
        "claim_frequency": np.random.poisson(2, 1000),
        "claim_severity": np.random.lognormal(mean=2, sigma=0.5, size=1000),
        "region": np.random.choice(["North", "South", "East", "West"], 1000),
        "policy_age": np.random.randint(1, 15, 1000)
    })

    reporting_analysis = ReportingAnalysis(sample_data)

    # Generate summary statistics
    summary = reporting_analysis.generate_summary_statistics()

    # Perform linear regression
    regression_results = reporting_analysis.linear_regression(dependent_var="claim_severity", 
                                                              independent_vars=["claim_frequency", "policy_age"])

    # Compute aggregate metrics
    aggregate_metrics = reporting_analysis.compute_aggregate_metrics(group_by_column="region")

if __name__ == "__main__":
    reporting_main()
# Step 5: Advanced Statistical Tests and Custom Reporting

from scipy.stats import chi2_contingency, mannwhitneyu

class StatisticalTests(ReportingAnalysis):
    def perform_chi_square_test(self, col1: str, col2: str):
        """
        Perform a Chi-Square test of independence between two categorical variables.

        Args:
            col1 (str): First categorical variable.
            col2 (str): Second categorical variable.

        Returns:
            dict: Test statistics and p-value.
        """
        contingency_table = pd.crosstab(self.data[col1], self.data[col2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-Square Test Results:\nChi2: {chi2}, p-value: {p}, Degrees of Freedom: {dof}")
        return {"chi2": chi2, "p-value": p, "dof": dof}

    def perform_mann_whitney_test(self, col1: str, col2: str):
        """
        Perform a Mann-Whitney U test for comparing two independent samples.

        Args:
            col1 (str): First sample data.
            col2 (str): Second sample data.

        Returns:
            dict: U statistic and p-value.
        """
        u_stat, p_value = mannwhitneyu(self.data[col1], self.data[col2])
        print(f"Mann-Whitney U Test Results:\nU Statistic: {u_stat}, p-value: {p_value}")
        return {"U statistic": u_stat, "p-value": p_value}

    def generate_custom_report(self, output_file: str):
        """
        Generate a custom HTML report summarizing key analyses.

        Args:
            output_file (str): Path to save the report.
        """
        summary = self.generate_summary_statistics()
        correlation_matrix = self.calculate_correlation_matrix()

        with open(output_file, 'w') as f:
            f.write("<html><head><title>Custom Report</title></head><body>")
            f.write("<h1>Summary Statistics</h1>")
            f.write(summary.to_html())
            f.write("<h1>Correlation Matrix</h1>")
            f.write(correlation_matrix.to_html())
            f.write("</body></html>")
        print(f"Report saved to {output_file}")

# Example usage for statistical tests and reporting
def statistical_tests_main():
    # Replace with actual dataset
    sample_data = pd.DataFrame({
        "claim_frequency": np.random.poisson(2, 1000),
        "claim_severity": np.random.lognormal(mean=2, sigma=0.5, size=1000),
        "region": np.random.choice(["North", "South", "East", "West"], 1000),
        "policy_type": np.random.choice(["Standard", "Premium"], 1000)
    })

    stats_tests = StatisticalTests(sample_data)

    # Chi-Square Test
    chi_square_results = stats_tests.perform_chi_square_test("region", "policy_type")
    print("Chi-Square Test Results:", chi_square_results)

    # Mann-Whitney Test
    mann_whitney_results = stats_tests.perform_mann_whitney_test("claim_frequency", "claim_severity")
    print("Mann-Whitney U Test Results:", mann_whitney_results)

    # Generate Custom Report
    stats_tests.generate_custom_report(output_file="custom_report.html")

if __name__ == "__main__":
    statistical_tests_main()









