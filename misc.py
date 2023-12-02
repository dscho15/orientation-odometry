def get_extractor(extractor_name: str):
    """
    Get feature extractor
    """
    if extractor_name == "sift":
        from sift_extractor import SiftExtractor
        return SiftExtractor()
    else:
        raise NotImplementedError("Extractor not implemented")
    
def get_matcher(matcher_name: str):
    """
    Get matcher
    """
    if matcher_name == "greedy":
        from matcher import GreedyMatcher
        return GreedyMatcher()
    elif matcher_name == "ratio":
        from matcher import RatioMatcher
        return RatioMatcher()
    else:
        raise NotImplementedError("Matcher not implemented")