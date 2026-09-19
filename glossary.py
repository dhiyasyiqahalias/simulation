import re

GLOSSARY = {
    "arima": {
        "aliases": ["arima", "arima model", "p d q", "pdq", "what is arima"],
        "explanation": "ARIMA (AutoRegressive Integrated Moving Average) is a statistical model used for time series forecasting. The parameters (p, d, q) represent the autoregressive terms, differencing required to make the series stationary, and moving average terms, respectively."
    },
    "confidence interval": {
        "aliases": ["confidence interval", "ci", "shaded area", "95%", "uncertainty band", "error band", "margin of error"],
        "explanation": "A confidence interval represents the range of values within which the actual future value is likely to fall. A 95% confidence interval means we can be 95% confident that the true future measurement will be inside this shaded area on the chart."
    },
    "forecast horizon": {
        "aliases": ["forecast horizon", "horizon", "years ahead", "prediction horizon"],
        "explanation": "The forecast horizon is the length of time into the future that the model is predicting. In this dashboard, you can adjust the horizon slider to predict up to several years ahead (e.g., up to 2028 or 2030)."
    },
    "k-means clustering": {
        "aliases": ["k-means", "kmeans", "clustering", "k means"],
        "explanation": "K-Means clustering is an unsupervised machine learning algorithm that groups data points into a defined number of clusters (in this case, 3). It groups Malaysian states based on similarities in their visitor density and visitor spending."
    },
    "tier classification": {
        "aliases": ["tier", "overcrowded", "balanced", "under-visited", "undervisited", "classification", "tiers"],
        "explanation": "Tiers are the categories assigned to states based on the K-Means model. 'Overcrowded' states face high visitor density and capacity strain. 'Balanced' states have a healthy visitor-to-resident ratio. 'Under-visited' states have low density and present growth/investment opportunities."
    },
    "density": {
        "aliases": ["density", "visitors per resident", "visitor density", "capacity strain"],
        "explanation": "Density measures the number of domestic visitors per local resident. A higher density indicates a greater strain on local infrastructure and resources, which contributes to an 'Overcrowded' classification."
    },
    "spend per visitor": {
        "aliases": ["spend per visitor", "spend", "expenditure", "yield", "economic yield", "receipts"],
        "explanation": "Spend per visitor represents the average amount of money (in RM) spent by a single domestic tourist. High spend per visitor indicates high economic yield from tourism."
    },
    "domestic visitors": {
        "aliases": ["domestic visitors", "tourists", "visitor volume", "visitor load"],
        "explanation": "The total number of local (domestic) tourists visiting a specific state or area within a given year."
    },
    "population": {
        "aliases": ["population", "resident population", "locals"],
        "explanation": "The total number of residents living in a state. Used as a baseline to calculate visitor density and assess capacity strain."
    },
    "labour force size": {
        "aliases": ["labour force size", "labor force", "workforce"],
        "explanation": "The total number of people in a state who are currently employed or actively seeking employment."
    },
    "employed": {
        "aliases": ["employed", "employment"],
        "explanation": "The total number of individuals in the labour force who currently hold a job."
    },
    "unemployed": {
        "aliases": ["unemployed", "unemployment", "unemployment rate"],
        "explanation": "The number of individuals in the labour force who do not have a job but are actively looking for one. The unemployment rate is this number represented as a percentage of the total labour force."
    },
    "employment population ratio": {
        "aliases": ["employment population ratio", "employment ratio"],
        "explanation": "The percentage of the total working-age population that is currently employed."
    },
    "mean income": {
        "aliases": ["mean income", "average income", "household income", "income"],
        "explanation": "The average gross monthly household income for residents in a specific state."
    },
    "gini": {
        "aliases": ["gini", "gini coefficient", "inequality", "gini index"],
        "explanation": "The Gini coefficient is a measure of statistical dispersion intended to represent income inequality within a state. A value of 0 represents perfect equality, while higher values indicate greater inequality."
    },
    "poverty rate": {
        "aliases": ["poverty rate", "poverty", "absolute poverty"],
        "explanation": "The percentage of households whose income falls below the absolute poverty line (PGK)."
    },
    "optimal model order": {
        "aliases": ["optimal model order", "model order", "best model", "aic", "lowest aic", "akaike"],
        "explanation": "The optimal model order refers to the specific (p,d,q) parameters chosen for the ARIMA model. The app automatically searches multiple combinations and selects the one with the lowest Akaike Information Criterion (AIC), balancing accuracy and statistical simplicity."
    },
    "trend": {
        "aliases": ["trend", "upward", "downward", "stable", "trajectory"],
        "explanation": "The projected direction of a metric over time based on the ARIMA forecast. An upward trend indicates growth, a downward trend indicates decline, and a stable trend indicates minimal change."
    }
}

def find_answer(user_question: str, glossary: dict = GLOSSARY) -> str:
    """
    Finds explanations for matched terms in a user's question.
    """
    # Normalize question (lowercase, strip punctuation)
    question_clean = re.sub(r'[^\w\s]', '', user_question).lower()
    
    matches = []
    for key, data in glossary.items():
        for alias in data["aliases"]:
            # Word boundary matching to avoid matching substrings incorrectly
            if re.search(r'\b' + re.escape(alias.lower()) + r'\b', question_clean):
                matches.append((key, data["explanation"]))
                break # Only add explanation once per term
    
    if matches:
        response = ""
        for key, explanation in matches:
            response += f"**{key.title()}**\n{explanation}\n\n"
        return response.strip()
    else:
        # Fallback message
        term_list = ", ".join([k.title() for k in list(glossary.keys())[:8]]) + ", etc."
        return f"I couldn't find that term in the glossary. Try asking about things like: {term_list}"
