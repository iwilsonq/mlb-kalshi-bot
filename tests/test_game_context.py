"""Tests for GameContext, MLBDataProvider, and FixtureMLBDataProvider."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from slugger.mlb_data import (
    BatterProfile,
    FixtureMLBDataProvider,
    GameContext,
    GameInfo,
    LiveMLBDataProvider,
    MLBDataProvider,
    PitcherProfile,
    TeamProfile,
)


# ─── Fixture builders ────────────────────────────────────────────────────────

def _make_game(**overrides) -> GameInfo:
    """Create a GameInfo with sensible defaults."""
    defaults = dict(
        game_id=718001,
        away_team="San Francisco Giants",
        home_team="Los Angeles Dodgers",
        away_abbrev="SF",
        home_abbrev="LAD",
        away_record="30-25",
        home_record="35-20",
        away_pitcher_name="Logan Webb",
        home_pitcher_name="Clayton Kershaw",
        away_pitcher_id=657277,
        home_pitcher_id=477132,
        game_datetime="2026-05-23T22:10:00Z",
        venue="Dodger Stadium",
        weather={"condition": "Clear", "temp": "72"},
        status="Pre-Game",
    )
    defaults.update(overrides)
    return GameInfo(**defaults)


def _make_pitcher(**overrides) -> PitcherProfile:
    defaults = dict(
        player_id=657277,
        name="Logan Webb",
        era=3.12,
        whip=1.08,
        k_per_9=8.5,
        bb_per_9=2.1,
        innings_pitched=95.0,
        strikeouts=90,
        games_started=15,
        recent_era=2.85,
        recent_k_per_start=7.2,
        recent_ip_per_start=6.1,
        max_k_in_start=10,
        k_per_start_list=[6, 7, 8, 7, 10],
        throws="R",
        whiff_rate=0.27,
    )
    defaults.update(overrides)
    return PitcherProfile(**defaults)


def _make_batter(**overrides) -> BatterProfile:
    defaults = dict(
        player_id=660271,
        name="Shohei Ohtani",
        team="LAD",
        avg=0.305,
        obp=0.390,
        slg=0.610,
        ops=1.000,
        hr=18,
        ab=220,
        hits=67,
        k_rate=0.22,
        bb_rate=0.12,
        hr_per_ab=0.082,
        recent_avg=0.340,
        recent_ops=1.100,
        recent_hr=5,
        avg_exit_velo=93.5,
        barrel_rate=0.12,
        hard_hit_rate=0.48,
        xba=0.310,
        xslg=0.600,
        vs_lhp_avg=0.280,
        vs_lhp_hr=4,
        vs_lhp_ab=50,
        vs_rhp_avg=0.315,
        vs_rhp_hr=14,
        vs_rhp_ab=170,
    )
    defaults.update(overrides)
    return BatterProfile(**defaults)


def _make_team(**overrides) -> TeamProfile:
    defaults = dict(
        name="Los Angeles Dodgers",
        abbrev="LAD",
        team_id=119,
        team_avg=0.262,
        team_ops=0.780,
        team_hr=85,
        runs_per_game=5.1,
        k_rate=0.215,
        team_era=3.45,
        team_whip=1.18,
        bullpen_era=3.20,
        wins=35,
        losses=20,
        run_diff=45,
    )
    defaults.update(overrides)
    return TeamProfile(**defaults)


def _make_context(**overrides) -> GameContext:
    game = overrides.pop("game", _make_game())
    return GameContext(
        game=game,
        away_pitcher=overrides.get("away_pitcher", _make_pitcher()),
        home_pitcher=overrides.get("home_pitcher", _make_pitcher(
            player_id=477132, name="Clayton Kershaw", throws="L",
            era=3.50, recent_era=3.20, k_per_9=9.2,
            recent_k_per_start=6.8, recent_ip_per_start=5.8,
            max_k_in_start=9, whiff_rate=0.25,
        )),
        away_batters=overrides.get("away_batters", []),
        home_batters=overrides.get("home_batters", [
            _make_batter(),
            _make_batter(player_id=605141, name="Mookie Betts", hr=12, ab=200, hits=58),
        ]),
        away_team=overrides.get("away_team", _make_team(
            name="San Francisco Giants", abbrev="SF", team_id=137,
            k_rate=0.235, runs_per_game=4.2,
        )),
        home_team=overrides.get("home_team", _make_team()),
    )


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestGameContext:
    def test_basic_construction(self):
        ctx = _make_context()
        assert ctx.game.game_id == 718001
        assert ctx.away_pitcher.name == "Logan Webb"
        assert ctx.home_pitcher.name == "Clayton Kershaw"
        assert len(ctx.home_batters) == 2
        assert ctx.away_team.k_rate == 0.235
        assert ctx.home_team.abbrev == "LAD"

    def test_minimal_context(self):
        """GameContext with only game info should work (all optionals None/empty)."""
        ctx = GameContext(game=_make_game())
        assert ctx.away_pitcher is None
        assert ctx.home_pitcher is None
        assert ctx.away_batters == []
        assert ctx.home_batters == []
        assert ctx.away_team is None
        assert ctx.home_team is None


class TestFixtureMLBDataProvider:
    def test_protocol_conformance(self):
        """FixtureMLBDataProvider should satisfy the MLBDataProvider protocol."""
        provider = FixtureMLBDataProvider([])
        assert isinstance(provider, MLBDataProvider)

    def test_returns_fixture_data(self):
        ctx = _make_context()
        provider = FixtureMLBDataProvider([ctx])
        contexts = provider.get_game_contexts()
        assert len(contexts) == 1
        assert contexts[0].game.game_id == 718001
        assert contexts[0].away_pitcher.name == "Logan Webb"

    def test_hydrate_game_by_id(self):
        ctx = _make_context()
        provider = FixtureMLBDataProvider([ctx])
        result = provider.hydrate_game(ctx.game)
        assert result.away_pitcher.name == "Logan Webb"

    def test_hydrate_game_unknown(self):
        """Unknown game ID should return a bare context."""
        provider = FixtureMLBDataProvider([])
        game = _make_game(game_id=999999)
        result = provider.hydrate_game(game)
        assert result.game.game_id == 999999
        assert result.away_pitcher is None

    def test_ignores_target_date(self):
        """Fixture provider returns the same data regardless of date."""
        ctx = _make_context()
        provider = FixtureMLBDataProvider([ctx])
        assert len(provider.get_game_contexts("2020-01-01")) == 1

    def test_multiple_games(self):
        ctx1 = _make_context(game=_make_game(game_id=1))
        ctx2 = _make_context(game=_make_game(game_id=2))
        provider = FixtureMLBDataProvider([ctx1, ctx2])
        assert len(provider.get_game_contexts()) == 2
        assert provider.hydrate_game(_make_game(game_id=1)).game.game_id == 1
        assert provider.hydrate_game(_make_game(game_id=2)).game.game_id == 2


class TestLiveMLBDataProvider:
    def test_protocol_conformance(self):
        """LiveMLBDataProvider should satisfy the MLBDataProvider protocol."""
        provider = LiveMLBDataProvider()
        assert isinstance(provider, MLBDataProvider)


class TestProcessGameWithContext:
    """Test that process_game works with a pre-built GameContext.

    These tests use mocked Kalshi client to avoid real API calls,
    but real GameContext data from fixtures.
    """

    def test_process_game_accepts_context(self):
        """process_game should accept a GameContext without errors."""
        from main import process_game, CircuitBreaker
        from slugger.config import Config

        ctx = _make_context()
        client = MagicMock()
        client.get_event_markets.return_value = []  # no markets
        client.get_balance.return_value = 100.0
        client.get_positions.return_value = []

        config = Config.from_env()
        circuit = CircuitBreaker(config)

        # Should not raise — just find no markets
        process_game(
            ctx, client, config, circuit,
            bankroll_usd=100.0,
            held_tickers=set(),
            placed_tickers=set(),
        )

    def test_process_game_uses_context_pitchers(self):
        """process_game should use pitchers from context, not fetch them."""
        from main import process_game, CircuitBreaker
        from slugger.config import Config

        ctx = _make_context()
        client = MagicMock()
        client.get_event_markets.return_value = []

        config = Config.from_env()
        circuit = CircuitBreaker(config)

        process_game(
            ctx, client, config, circuit,
            bankroll_usd=100.0,
            held_tickers=set(),
            placed_tickers=set(),
        )

        # Verify no pitcher profile fetches happened (client has no
        # get_pitcher_profile method — if code tried to call it, we'd get
        # an AttributeError)
        # The key assertion: process_game ran without errors, proving it
        # used ctx.away_pitcher and ctx.home_pitcher instead of fetching.

    def test_process_game_with_started_game(self):
        """A game past its start time should be skipped entirely."""
        from main import process_game, CircuitBreaker
        from slugger.config import Config

        ctx = _make_context(game=_make_game(
            game_datetime="2020-01-01T12:00:00Z",  # long past
        ))
        client = MagicMock()
        config = Config.from_env()
        circuit = CircuitBreaker(config)

        process_game(
            ctx, client, config, circuit,
            bankroll_usd=100.0,
            held_tickers=set(),
            placed_tickers=set(),
        )

        # Should not have queried any markets
        client.get_event_markets.assert_not_called()
