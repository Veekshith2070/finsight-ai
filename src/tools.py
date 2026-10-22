from src.logger import logger

def calculate_percentage_growth(old_value: float, new_value: float) -> dict:
    """
    Calculates the percentage gowth between two financial periods:
    Growth (%) = ((new_value - old_value) / abs(old_value))*100

    Args:
         old_value: The base or prior period value (e.g. prior year revenue).
         new_value: The current period value.
    """

    logger.info(f"Tool executed: calculate_percentage_growth(old={old_value}, new={new_value})")

    if old_value == 0:
        return {"error": "Cannot claculate percentage growth from a base value zero."}

    growth = ((new_value - old_value) / abs(old_value))*100
    
    return {
        "old_value": old_value,
        "new_value": new_value,
        "percentage_growth": round(growth, 2),
        "formatted": f"{growth:+.2f}"
    }

def calculate_financial_margin(numerator: float, denominator: float, margin_name: str = "Margin") -> dict:
    """
    Calculates a financial margin (e.g., Gross Margin %, Operating Margin %, Net Margin %):
    Margin (%) = (numerator / denominator) * 100

    Args:
       numerator: The profit figure (e.g., Gross Profit, Operating Income, Net Income).
       denominator: The total revenue or net sales figure.
       margin_name: The name of the margin being calculated.
    """
    logger.info(f"Tool executed: calculate_financial_margin(num={numerator}, den={denominator})")

    if denominator == 0:
        return {"error": "Cannot calculate margin with revenue/denominator of zero."}

    margin = (numerator/denominator)*100

    return {
        "margin_name": margin_name,
        "margin_percentage": round(margin, 2),
        "formatted": f"{margin:.2f}%"
    }

def calculate_financial_ratio(numerator: float, denominator: float, ratio_name: str = "Ratio") -> dict:
    """
    Calculates a standard financial ratio (e.g., Debt-to-Equity, Current Ratio):
    Ratio = numerator/denominator
    """

    logger.info(f"Tool executed: claculate_financial_ratio(num={numerator}, den={denominator})")

    if denominator == 0:
        return {"error": "Cannot calculate ratio with denominator zero."}

    ratio = numerator/denominator

    return {
        "ratio_name": ratio_name,
        "ratio_value": round(ratio, 4),
        "formatted": f"{ratio:.2f}x"
    }

def calculate_financial_difference(
    value_a: float,
    value_b: float,
    metric_name: str = "Difference / Gross Profit"
) -> dict:
    """
    Calculates the exact arithmetic difference between two financial values (value_a - value_b).
    Useful for calculating Gross Profit (Revenue - Cost of Sales) or Net Variance.
    """
    diff = round(value_a - value_b, 2)
    logger.info(f"Tool executed: calculate_financial_difference({value_a} - {value_b} = {diff})")
    return {
        "metric_name": metric_name,
        "value_a": value_a,
        "value_b": value_b,
        "difference": diff,
        "formatted": f"${diff:,.2f} million" if diff > 1000 else f"${diff:,.2f}"
    }
# A list of available financial tools
FINANCIAL_TOOLS = {
    calculate_financial_margin,
    calculate_percentage_growth,
    calculate_financial_ratio,
    calculate_financial_difference
}

if __name__ == "__main__":
    # Test 1: Apple Services growth from $27,423M to $30,739M
    growth_test = calculate_percentage_growth(27423.0, 30739.0)
    print("Test 1 (Growth):", growth_test)
    
    # Test 2: Aplle Operating margin: Operating Income ($35,695M) / Total Net Sales ($109,417M)
    margin_test = calculate_financial_margin(35695.0, 109417.0, margin_name="Operating Margin")
    print("Test 2 (Operating Margin):", margin_test)