def get_recommendation(risk_level):
    """
    Maps the Risk Category to a set of recommended actions
    based on the NHAA (14566) problem statement.
    """
    recommendations = {
        "Low": {
            "action": "Standard Support",
            "details": "Provide general information, resource guides, and standard grievance redressal tracking.",
            "priority": "Routine"
        },
        "Moderate": {
            "action": "Prioritized Counselling",
            "details": "Schedule a session with a certified counsellor and provide emotional support resources.",
            "priority": "Medium"
        },
        "High": {
            "action": "Urgent Intervention",
            "details": "Immediately refer to legal aid for SC/ST Act assistance and coordinate with medical professionals for trauma care.",
            "priority": "High"
        },
        "Critical": {
            "action": "Emergency Response",
            "details": "Immediate police intervention required. Activate witness protection protocols and emergency medical support. Notify District Administration.",
            "priority": "Urgent"
        }
    }

    return recommendations.get(risk_level, recommendations["Low"])

def map_svi_to_category(svi):
    """
    Maps the Stress Vulnerability Index (SVI) score (0.0 - 1.0)
    to a risk category.
    """
    if svi >= 0.8:
        return "Critical"
    elif svi >= 0.5:
        return "High"
    elif svi >= 0.3:
        return "Moderate"
    else:
        return "Low"
