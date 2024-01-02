def get_extractor(extractor_name: str):
    if extractor_name == "sift":
        from feature_extractor import SiftExtractor

        return SiftExtractor()
    elif extractor_name == "root_sift":
        from feature_extractor import RootSiftExtractor

        return RootSiftExtractor()
    elif extractor_name == "orb":
        from feature_extractor import OrbExtractor

        return OrbExtractor()
    else:
        raise NotImplementedError("Extractor not implemented")


def get_matcher(matcher_name: str):
    if matcher_name == "greedy":
        from matcher import GreedyMatcher

        return GreedyMatcher()
    elif matcher_name == "ratio":
        from matcher import RatioMatcher

        return RatioMatcher()
    else:
        raise NotImplementedError("Matcher not implemented")
