"""In-game MLB win probability (WP) model — fair-value anchor for trading.

Usage:
    from slugger.wp import WPModel, get_wp
    wp = get_wp({"inning": 7, "is_top": False, "outs": 1,
                 "score_diff": 2, "on1": True, "on2": False, "on3": False})
"""
from slugger.wp.model import WPModel, get_wp, clear_wp_model_cache

__all__ = ["WPModel", "get_wp", "clear_wp_model_cache"]
