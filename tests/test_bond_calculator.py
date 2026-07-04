"""Tests for bond_calculator core functions."""

import pytest
from core.bond_calculator import (
    conversion_value,
    premium_rate,
    double_low_value,
    redemption_trigger_ratio,
    putback_trigger_ratio,
    estimate_remaining_years,
    estimate_discount_rate,
    estimate_coupon_schedule,
)


class TestConversionValue:
    def test_normal(self):
        assert conversion_value(10.0, 12.0) == pytest.approx(120.0)

    def test_zero_conversion_price(self):
        import math
        assert math.isnan(conversion_value(0, 12.0))

    def test_low_stock_price(self):
        cv = conversion_value(20.0, 5.0)
        assert cv == pytest.approx(25.0)


class TestPremiumRate:
    def test_positive_premium(self):
        pr = premium_rate(120, 100)
        assert pr == pytest.approx(20.0)

    def test_negative_premium(self):
        pr = premium_rate(95, 100)
        assert pr == pytest.approx(-5.0)

    def test_discount(self):
        pr = premium_rate(80, 100)
        assert pr == pytest.approx(-20.0)


class TestDoubleLow:
    def test_normal(self):
        dl = double_low_value(115, 15)
        assert dl == pytest.approx(130.0)  # 115 + 15

    def test_negative_premium_ignored(self):
        dl = double_low_value(100, -5)
        assert dl == pytest.approx(100.0)


class TestRedemptionRatio:
    def test_normal(self):
        assert redemption_trigger_ratio(13, 10) == pytest.approx(1.3)

    def test_below_threshold(self):
        assert redemption_trigger_ratio(10, 10) == pytest.approx(1.0)


class TestPutbackRatio:
    def test_normal(self):
        # putback = stock / (conversion * 0.7)
        r = putback_trigger_ratio(7, 10)
        assert r == pytest.approx(1.0)


class TestEstimateRemainingYears:
    def test_normal(self):
        r = estimate_remaining_years("20280615", "20260615")
        assert 1.9 < r < 2.1

    def test_different_format(self):
        r = estimate_remaining_years("2028-06-15", "2026-06-15")
        assert 1.9 < r < 2.1


class TestDiscountRate:
    def test_aaa(self):
        assert estimate_discount_rate("AAA") == pytest.approx(0.03)

    def test_aa(self):
        assert estimate_discount_rate("AA") == pytest.approx(0.04)

    def test_unknown(self):
        assert estimate_discount_rate("B-") > 0.05


class TestEstimateCouponSchedule:
    def test_aaa_full_schedule(self):
        coupons = estimate_coupon_schedule("AAA")
        assert len(coupons) == 6
        assert coupons[0] == 0.002
        assert coupons[-1] == 0.020
        # Should be stepped (increasing)
        for i in range(len(coupons) - 1):
            assert coupons[i] <= coupons[i + 1]

    def test_aa_full_schedule(self):
        coupons = estimate_coupon_schedule("AA")
        assert len(coupons) == 6
        assert coupons[0] == 0.003

    def test_a_plus_schedule(self):
        coupons = estimate_coupon_schedule("A+")
        assert coupons[0] == 0.005

    def test_low_rating_schedule(self):
        coupons = estimate_coupon_schedule("BBB")
        assert coupons[0] == 0.008
        assert coupons[-1] == 0.030

    def test_remaining_years_subset(self):
        # 3 years remaining should return last 3 coupons (higher late-stage rates)
        coupons = estimate_coupon_schedule("AA", remaining_years=3)
        assert len(coupons) == 3
        assert coupons[0] == 0.012  # year 4 rate

    def test_remaining_years_exceeds_total(self):
        # More remaining years than total → clamped to full schedule
        coupons = estimate_coupon_schedule("AA", total_years=6, remaining_years=8)
        assert len(coupons) == 6  # clamped to total_years

    def test_custom_total_years(self):
        coupons = estimate_coupon_schedule("AA", total_years=5)
        assert len(coupons) == 5
        assert coupons[-1] == 0.015
