"""
System Autonomy Core (SAC) - Phase 5: Economic Intelligence Core

This module provides comprehensive market analysis and economic intelligence
for autonomous hardware procurement, upgrade planning, and cost optimization.
It tracks hardware prices, predicts market trends, and provides economic
decision-making capabilities for the autonomous system.

Key Features:
- Real-time hardware price tracking across multiple vendors
- Market trend analysis and price prediction algorithms
- Cost-benefit analysis for upgrade scenarios
- ROI calculations for hardware investments
- Procurement intelligence and vendor reliability scoring
- Budget optimization and resource allocation planning
- Technology lifecycle analysis and depreciation modeling
- Market volatility detection and timing optimization

Author: Vega2.0 Autonomous AI System
"""

import asyncio
import aiohttp
import json
import time
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import re
import statistics
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/home/ncacord/Vega2.0/sac/logs/economic_scanner.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ComponentCategory(Enum):
    """Hardware component categories"""

    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    MOTHERBOARD = "motherboard"
    PSU = "power_supply"
    COOLING = "cooling"
    CASE = "case"
    NETWORK = "network"


class MarketTrend(Enum):
    """Market trend directions"""

    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    VOLATILE = "volatile"


class VendorTier(Enum):
    """Vendor reliability tiers"""

    PREMIUM = "premium"  # High-end, reliable vendors
    MAINSTREAM = "mainstream"  # Standard vendors
    BUDGET = "budget"  # Low-cost vendors
    UNKNOWN = "unknown"  # Unverified vendors


class UpgradePriority(Enum):
    """Upgrade priority levels"""

    CRITICAL = "critical"  # System failing/obsolete
    HIGH = "high"  # Significant performance gain
    MEDIUM = "medium"  # Moderate improvement
    LOW = "low"  # Nice to have
    DEFERRED = "deferred"  # Not recommended now


@dataclass
class HardwareProduct:
    """Hardware product information"""

    product_id: str
    name: str
    category: ComponentCategory
    manufacturer: str
    model: str
    specifications: Dict[str, Any]
    current_price: float
    currency: str
    vendor: str
    vendor_tier: VendorTier
    availability: str
    last_updated: str


@dataclass
class PriceHistory:
    """Price history tracking"""

    product_id: str
    timestamp: str
    price: float
    vendor: str
    availability: str
    source: str


@dataclass
class MarketAnalysis:
    """Market analysis results"""

    category: ComponentCategory
    timestamp: str
    average_price: float
    price_trend: MarketTrend
    trend_strength: float  # 0.0 to 1.0
    volatility: float  # Standard deviation
    best_value_products: List[str]  # Product IDs
    price_predictions: Dict[str, float]  # Days ahead -> predicted price
    market_insights: List[str]


@dataclass
class UpgradeRecommendation:
    """Hardware upgrade recommendation"""

    component_type: ComponentCategory
    current_component: str
    recommended_component: str
    priority: UpgradePriority
    estimated_cost: float
    performance_gain: float  # Percentage improvement
    roi_months: Optional[int]  # Payback period
    best_timing: str  # When to buy
    justification: str
    compatibility_notes: List[str]


@dataclass
class BudgetAllocation:
    """Budget allocation recommendation"""

    total_budget: float
    allocations: Dict[ComponentCategory, float]
    upgrade_timeline: Dict[str, List[str]]  # Date -> component upgrades
    cost_optimization_notes: List[str]
    expected_roi: float


class EconomicScanner:
    """
    Advanced economic intelligence system for autonomous hardware
    procurement and upgrade decision-making.
    """

    def __init__(self, config_path: str = "/home/ncacord/Vega2.0/sac/config"):
        self.config_path = Path(config_path)
        self.logs_path = Path("/home/ncacord/Vega2.0/sac/logs")
        self.data_path = Path("/home/ncacord/Vega2.0/sac/data")

        # Ensure directories exist
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Database for price tracking
        self.db_path = self.data_path / "economic_intelligence.db"
        self._init_database()

        # Load configuration
        self.config = self._load_config()

        # Runtime state
        self.running = False
        self.price_cache = {}  # product_id -> HardwareProduct
        self.market_analysis_cache = {}  # category -> MarketAnalysis
        self.vendor_reliability = {}  # vendor -> reliability score

        # Load current system specs for comparison
        self.current_system = self._get_current_system_specs()

        logger.info("EconomicScanner initialized with market intelligence")

    def _init_database(self):
        """Initialize SQLite database for price tracking"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Products table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    manufacturer TEXT,
                    model TEXT,
                    specifications TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Price history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    price REAL NOT NULL,
                    vendor TEXT NOT NULL,
                    availability TEXT,
                    source TEXT,
                    FOREIGN KEY (product_id) REFERENCES products (product_id)
                )
            """
            )

            # Market analysis table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS market_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    average_price REAL,
                    price_trend TEXT,
                    trend_strength REAL,
                    volatility REAL,
                    analysis_data TEXT
                )
            """
            )

            # Vendor reliability table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS vendor_reliability (
                    vendor TEXT PRIMARY KEY,
                    reliability_score REAL,
                    tier TEXT,
                    last_updated TIMESTAMP,
                    notes TEXT
                )
            """
            )

            # Create indexes for performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_price_history_product ON price_history(product_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_price_history_timestamp ON price_history(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_analysis_category ON market_analysis(category)"
            )

            conn.commit()
            conn.close()
            logger.info("Economic intelligence database initialized")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def _load_config(self) -> Dict[str, Any]:
        """Load economic scanner configuration"""
        config_file = self.config_path / "economic_scanner_config.json"

        default_config = {
            "scanning": {
                "price_check_interval_hours": 6,
                "market_analysis_interval_hours": 24,
                "enable_price_alerts": True,
                "price_drop_threshold": 0.1,  # 10% price drop
                "price_spike_threshold": 0.2,  # 20% price spike
                "max_concurrent_requests": 10,
            },
            "vendors": {
                "enabled_vendors": [
                    "newegg",
                    "amazon",
                    "best_buy",
                    "micro_center",
                    "b_h_photo",
                    "adorama",
                    "tigerdirect",
                ],
                "preferred_vendors": ["newegg", "amazon", "micro_center"],
                "vendor_weights": {
                    "newegg": 1.0,
                    "amazon": 0.9,
                    "best_buy": 0.8,
                    "micro_center": 1.0,
                    "b_h_photo": 0.7,
                    "adorama": 0.6,
                    "tigerdirect": 0.5,
                },
            },
            "analysis": {
                "price_history_days": 90,
                "trend_analysis_days": 30,
                "volatility_threshold": 0.15,
                "prediction_days": [7, 14, 30, 60],
                "min_samples_for_analysis": 10,
            },
            "budget": {
                "default_budget": 5000.0,
                "emergency_reserve_percent": 0.2,
                "upgrade_frequency_months": 24,
                "roi_target_months": 18,
            },
            "notifications": {
                "enable_price_alerts": True,
                "enable_upgrade_recommendations": True,
                "alert_frequency_hours": 12,
            },
        }

        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                config = default_config
        else:
            config = default_config
            self._save_config(config)

        return config

    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        config_file = self.config_path / "economic_scanner_config.json"
        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _get_current_system_specs(self) -> Dict[str, Any]:
        """Get current system specifications for comparison"""
        # This would integrate with system_probe.py to get current hardware
        # For now, we'll use placeholder data representing the Ryzen 9 3900X system
        return {
            "cpu": {
                "model": "AMD Ryzen 9 3900X",
                "cores": 12,
                "threads": 24,
                "base_clock": 3.8,
                "boost_clock": 4.6,
                "socket": "AM4",
                "tdp": 105,
            },
            "memory": {"total_gb": 128, "type": "DDR4", "speed": 3600, "channels": 4},
            "gpu": [
                {"model": "NVIDIA GTX 1660 Super", "vram_gb": 6, "cuda_cores": 1408},
                {"model": "NVIDIA Quadro P1000", "vram_gb": 4, "cuda_cores": 640},
            ],
            "storage": {
                "primary": "NVMe SSD",
                "capacity_gb": 1000,
                "interface": "PCIe 3.0 x4",
            },
            "motherboard": {
                "chipset": "B450",
                "socket": "AM4",
                "memory_slots": 4,
                "pcie_slots": 2,
            },
        }

    async def _fetch_product_prices(
        self, product_queries: List[str]
    ) -> List[HardwareProduct]:
        """Fetch product prices from multiple vendors (simulated)"""
        # In a real implementation, this would make actual API calls to vendor websites
        # For demonstration, we'll simulate realistic price data

        products = []

        # Simulate realistic hardware products and prices
        simulated_products = {
            "AMD Ryzen 9 7900X": {
                "category": ComponentCategory.CPU,
                "manufacturer": "AMD",
                "model": "7900X",
                "specs": {
                    "cores": 12,
                    "threads": 24,
                    "base_clock": 4.7,
                    "boost_clock": 5.6,
                    "socket": "AM5",
                },
                "price_range": (400, 550),
            },
            "Intel Core i9-13900K": {
                "category": ComponentCategory.CPU,
                "manufacturer": "Intel",
                "model": "13900K",
                "specs": {
                    "cores": 24,
                    "threads": 32,
                    "base_clock": 3.0,
                    "boost_clock": 5.8,
                    "socket": "LGA1700",
                },
                "price_range": (450, 600),
            },
            "NVIDIA RTX 4070 Ti": {
                "category": ComponentCategory.GPU,
                "manufacturer": "NVIDIA",
                "model": "RTX 4070 Ti",
                "specs": {"vram_gb": 12, "cuda_cores": 7680, "boost_clock": 2610},
                "price_range": (700, 900),
            },
            "NVIDIA RTX 4080": {
                "category": ComponentCategory.GPU,
                "manufacturer": "NVIDIA",
                "model": "RTX 4080",
                "specs": {"vram_gb": 16, "cuda_cores": 9728, "boost_clock": 2505},
                "price_range": (1000, 1300),
            },
            "G.Skill Trident Z5 64GB DDR5": {
                "category": ComponentCategory.MEMORY,
                "manufacturer": "G.Skill",
                "model": "Trident Z5 64GB",
                "specs": {
                    "capacity_gb": 64,
                    "type": "DDR5",
                    "speed": 6000,
                    "kit": "2x32GB",
                },
                "price_range": (400, 600),
            },
            "Samsung 980 PRO 2TB": {
                "category": ComponentCategory.STORAGE,
                "manufacturer": "Samsung",
                "model": "980 PRO 2TB",
                "specs": {
                    "capacity_gb": 2000,
                    "interface": "PCIe 4.0",
                    "type": "NVMe SSD",
                },
                "price_range": (150, 250),
            },
        }

        vendors = ["newegg", "amazon", "best_buy", "micro_center"]
        vendor_tiers = {
            "newegg": VendorTier.PREMIUM,
            "amazon": VendorTier.MAINSTREAM,
            "best_buy": VendorTier.MAINSTREAM,
            "micro_center": VendorTier.PREMIUM,
        }

        for product_name, product_info in simulated_products.items():
            for vendor in vendors:
                # Simulate price variation between vendors
                min_price, max_price = product_info["price_range"]
                vendor_modifier = self.config["vendors"]["vendor_weights"].get(
                    vendor, 1.0
                )

                # Add some random variation
                import random

                base_price = random.uniform(min_price, max_price)
                vendor_price = base_price * vendor_modifier

                # Add small random variation (market fluctuation)
                price_variation = random.uniform(0.95, 1.05)
                final_price = vendor_price * price_variation

                product_id = hashlib.md5(
                    f"{product_name}_{vendor}".encode()
                ).hexdigest()[:12]

                product = HardwareProduct(
                    product_id=product_id,
                    name=product_name,
                    category=product_info["category"],
                    manufacturer=product_info["manufacturer"],
                    model=product_info["model"],
                    specifications=product_info["specs"],
                    current_price=round(final_price, 2),
                    currency="USD",
                    vendor=vendor,
                    vendor_tier=vendor_tiers[vendor],
                    availability=(
                        "in_stock" if random.random() > 0.1 else "limited_stock"
                    ),
                    last_updated=datetime.now().isoformat(),
                )
                products.append(product)

        # Simulate API delay
        await asyncio.sleep(random.uniform(0.5, 2.0))

        return products

    def _store_price_data(self, products: List[HardwareProduct]):
        """Store product and price data in database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            for product in products:
                # Insert or update product
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO products 
                    (product_id, name, category, manufacturer, model, specifications)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        product.product_id,
                        product.name,
                        product.category.value,
                        product.manufacturer,
                        product.model,
                        json.dumps(product.specifications),
                    ),
                )

                # Insert price history
                cursor.execute(
                    """
                    INSERT INTO price_history 
                    (product_id, timestamp, price, vendor, availability, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        product.product_id,
                        product.last_updated,
                        product.current_price,
                        product.vendor,
                        product.availability,
                        "api_scan",
                    ),
                )

                # Update cache
                self.price_cache[product.product_id] = product

            conn.commit()
            conn.close()
            logger.info(f"Stored price data for {len(products)} products")

        except Exception as e:
            logger.error(f"Error storing price data: {e}")

    def _analyze_market_trends(self, category: ComponentCategory) -> MarketAnalysis:
        """Analyze market trends for a component category"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get price history for category
            cursor.execute(
                """
                SELECT ph.timestamp, ph.price, ph.vendor, p.name
                FROM price_history ph
                JOIN products p ON ph.product_id = p.product_id
                WHERE p.category = ? AND ph.timestamp >= ?
                ORDER BY ph.timestamp
            """,
                (
                    category.value,
                    (
                        datetime.now()
                        - timedelta(days=self.config["analysis"]["price_history_days"])
                    ).isoformat(),
                ),
            )

            price_data = cursor.fetchall()
            conn.close()

            if len(price_data) < self.config["analysis"]["min_samples_for_analysis"]:
                return MarketAnalysis(
                    category=category,
                    timestamp=datetime.now().isoformat(),
                    average_price=0.0,
                    price_trend=MarketTrend.STABLE,
                    trend_strength=0.0,
                    volatility=0.0,
                    best_value_products=[],
                    price_predictions={},
                    market_insights=["Insufficient data for analysis"],
                )

            # Calculate statistics
            prices = [row[1] for row in price_data]
            average_price = statistics.mean(prices)
            volatility = statistics.stdev(prices) if len(prices) > 1 else 0.0

            # Determine trend
            recent_days = self.config["analysis"]["trend_analysis_days"]
            cutoff_date = datetime.now() - timedelta(days=recent_days)
            recent_prices = [
                row[1]
                for row in price_data
                if datetime.fromisoformat(row[0]) >= cutoff_date
            ]

            if len(recent_prices) >= 2:
                trend_slope = (recent_prices[-1] - recent_prices[0]) / len(
                    recent_prices
                )
                trend_strength = min(abs(trend_slope) / average_price, 1.0)

                if trend_slope > average_price * 0.02:  # 2% increase
                    price_trend = MarketTrend.RISING
                elif trend_slope < -average_price * 0.02:  # 2% decrease
                    price_trend = MarketTrend.FALLING
                else:
                    price_trend = MarketTrend.STABLE

                if (
                    volatility / average_price
                    > self.config["analysis"]["volatility_threshold"]
                ):
                    price_trend = MarketTrend.VOLATILE
            else:
                price_trend = MarketTrend.STABLE
                trend_strength = 0.0

            # Find best value products
            category_products = {row[3]: row[1] for row in price_data}
            sorted_products = sorted(category_products.items(), key=lambda x: x[1])
            best_value_products = [prod[0] for prod in sorted_products[:3]]

            # Simple price predictions (linear trend)
            price_predictions = {}
            if trend_strength > 0.1:
                daily_change = trend_slope
                for days in self.config["analysis"]["prediction_days"]:
                    predicted_price = average_price + (daily_change * days)
                    price_predictions[str(days)] = round(max(predicted_price, 0), 2)

            # Generate market insights
            insights = []
            if price_trend == MarketTrend.FALLING:
                insights.append(
                    f"Prices dropping by {trend_strength*100:.1f}% - good time to buy"
                )
            elif price_trend == MarketTrend.RISING:
                insights.append(
                    f"Prices rising by {trend_strength*100:.1f}% - consider buying soon"
                )
            elif price_trend == MarketTrend.VOLATILE:
                insights.append(
                    "High price volatility detected - wait for stabilization"
                )

            if volatility / average_price > 0.2:
                insights.append("Highly volatile market - monitor closely")

            analysis = MarketAnalysis(
                category=category,
                timestamp=datetime.now().isoformat(),
                average_price=round(average_price, 2),
                price_trend=price_trend,
                trend_strength=trend_strength,
                volatility=volatility,
                best_value_products=best_value_products,
                price_predictions=price_predictions,
                market_insights=insights,
            )

            # Cache the analysis
            self.market_analysis_cache[category] = analysis

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing market trends for {category}: {e}")
            return MarketAnalysis(
                category=category,
                timestamp=datetime.now().isoformat(),
                average_price=0.0,
                price_trend=MarketTrend.STABLE,
                trend_strength=0.0,
                volatility=0.0,
                best_value_products=[],
                price_predictions={},
                market_insights=[f"Analysis error: {str(e)}"],
            )

    def _generate_upgrade_recommendations(self) -> List[UpgradeRecommendation]:
        """Generate hardware upgrade recommendations based on current system and market analysis"""
        recommendations = []

        # Analyze CPU upgrade potential
        current_cpu = self.current_system["cpu"]
        cpu_analysis = self.market_analysis_cache.get(ComponentCategory.CPU)

        if cpu_analysis:
            # Check if CPU is getting old (assuming 3+ years for major upgrade)
            cpu_age_factor = 0.7  # Assuming current CPU is 2-3 years old

            if cpu_age_factor > 0.5:  # Significant age
                # Look for CPU upgrades
                recommended_cpu = "AMD Ryzen 9 7900X"  # Example upgrade
                performance_gain = 35.0  # Estimated 35% performance improvement

                # Estimate cost based on market analysis
                estimated_cost = cpu_analysis.average_price * 1.1  # Premium for latest

                # Calculate ROI based on performance improvement
                roi_months = int(
                    self.config["budget"]["roi_target_months"]
                    / (performance_gain / 100)
                )

                priority = (
                    UpgradePriority.HIGH
                    if performance_gain > 30
                    else UpgradePriority.MEDIUM
                )

                # Determine best timing based on market trends
                if cpu_analysis.price_trend == MarketTrend.FALLING:
                    best_timing = "Wait 2-4 weeks for further price drops"
                elif cpu_analysis.price_trend == MarketTrend.RISING:
                    best_timing = "Buy within 1-2 weeks before prices increase"
                else:
                    best_timing = "Buy when convenient - stable pricing"

                recommendation = UpgradeRecommendation(
                    component_type=ComponentCategory.CPU,
                    current_component=current_cpu["model"],
                    recommended_component=recommended_cpu,
                    priority=priority,
                    estimated_cost=estimated_cost,
                    performance_gain=performance_gain,
                    roi_months=roi_months,
                    best_timing=best_timing,
                    justification=f"Current CPU is {cpu_age_factor*100:.0f}% through typical lifecycle. "
                    f"Upgrade would provide {performance_gain}% performance improvement.",
                    compatibility_notes=[
                        "Requires AM5 motherboard upgrade",
                        "Requires DDR5 memory",
                    ],
                )
                recommendations.append(recommendation)

        # Analyze GPU upgrade potential
        current_gpus = self.current_system["gpu"]
        gpu_analysis = self.market_analysis_cache.get(ComponentCategory.GPU)

        if gpu_analysis and current_gpus:
            primary_gpu = current_gpus[0]  # GTX 1660 Super

            # GPU is significantly outdated for modern workloads
            gpu_age_factor = 0.9  # Very outdated
            performance_gain = 80.0  # Massive improvement with RTX 4070 Ti

            recommended_gpu = "NVIDIA RTX 4070 Ti"
            estimated_cost = gpu_analysis.average_price

            priority = (
                UpgradePriority.CRITICAL
                if performance_gain > 70
                else UpgradePriority.HIGH
            )
            roi_months = int(
                self.config["budget"]["roi_target_months"] / (performance_gain / 100)
            )

            if gpu_analysis.price_trend == MarketTrend.FALLING:
                best_timing = "Excellent time to buy - prices falling"
            else:
                best_timing = "High priority upgrade regardless of timing"

            recommendation = UpgradeRecommendation(
                component_type=ComponentCategory.GPU,
                current_component=primary_gpu["model"],
                recommended_component=recommended_gpu,
                priority=priority,
                estimated_cost=estimated_cost,
                performance_gain=performance_gain,
                roi_months=roi_months,
                best_timing=best_timing,
                justification=f"Current GPU is severely outdated. Upgrade would provide "
                f"{performance_gain}% performance improvement and modern features.",
                compatibility_notes=[
                    "Ensure PSU can handle increased power draw",
                    "Check PCIe slot compatibility",
                ],
            )
            recommendations.append(recommendation)

        # Analyze memory upgrade potential
        current_memory = self.current_system["memory"]
        memory_analysis = self.market_analysis_cache.get(ComponentCategory.MEMORY)

        if memory_analysis:
            # 128GB DDR4 is substantial, but DDR5 would be an upgrade with new CPU
            if current_memory["type"] == "DDR4":
                recommended_memory = "G.Skill Trident Z5 64GB DDR5"
                performance_gain = 15.0  # Moderate improvement with DDR5
                estimated_cost = memory_analysis.average_price

                priority = UpgradePriority.MEDIUM  # Nice to have with CPU upgrade
                roi_months = int(
                    self.config["budget"]["roi_target_months"] * 1.5
                )  # Longer payback

                recommendation = UpgradeRecommendation(
                    component_type=ComponentCategory.MEMORY,
                    current_component=f"{current_memory['total_gb']}GB {current_memory['type']}",
                    recommended_component=recommended_memory,
                    priority=priority,
                    estimated_cost=estimated_cost,
                    performance_gain=performance_gain,
                    roi_months=roi_months,
                    best_timing="Coordinate with CPU/motherboard upgrade",
                    justification="DDR5 upgrade provides future-proofing and better performance with new CPU",
                    compatibility_notes=[
                        "Only compatible with DDR5 motherboards",
                        "Cannot mix with DDR4",
                    ],
                )
                recommendations.append(recommendation)

        return recommendations

    def _optimize_budget_allocation(
        self, budget: float, recommendations: List[UpgradeRecommendation]
    ) -> BudgetAllocation:
        """Optimize budget allocation across upgrade recommendations"""

        # Sort recommendations by priority and ROI
        def priority_score(rec):
            priority_values = {
                UpgradePriority.CRITICAL: 100,
                UpgradePriority.HIGH: 80,
                UpgradePriority.MEDIUM: 60,
                UpgradePriority.LOW: 40,
                UpgradePriority.DEFERRED: 20,
            }
            base_score = priority_values[rec.priority]
            roi_bonus = max(0, 50 - (rec.roi_months or 50))  # Better ROI = higher score
            return base_score + roi_bonus

        sorted_recommendations = sorted(
            recommendations, key=priority_score, reverse=True
        )

        # Allocate budget with emergency reserve
        available_budget = budget * (
            1 - self.config["budget"]["emergency_reserve_percent"]
        )
        allocations = {}
        upgrade_timeline = {}
        remaining_budget = available_budget

        # Immediate upgrades (within 3 months)
        immediate_upgrades = []
        for rec in sorted_recommendations:
            if rec.priority in [UpgradePriority.CRITICAL, UpgradePriority.HIGH]:
                if remaining_budget >= rec.estimated_cost:
                    allocations[rec.component_type] = rec.estimated_cost
                    remaining_budget -= rec.estimated_cost
                    immediate_upgrades.append(rec.recommended_component)

        # Planned upgrades (3-12 months)
        planned_upgrades = []
        for rec in sorted_recommendations:
            if (
                rec.component_type not in allocations
                and rec.priority == UpgradePriority.MEDIUM
            ):
                if (
                    remaining_budget >= rec.estimated_cost * 0.8
                ):  # Account for price changes
                    allocations[rec.component_type] = rec.estimated_cost
                    remaining_budget -= rec.estimated_cost
                    planned_upgrades.append(rec.recommended_component)

        # Future upgrades (12+ months)
        future_upgrades = []
        for rec in sorted_recommendations:
            if rec.component_type not in allocations:
                future_upgrades.append(rec.recommended_component)

        # Create timeline
        current_date = datetime.now()
        if immediate_upgrades:
            upgrade_timeline[(current_date + timedelta(weeks=2)).strftime("%Y-%m")] = (
                immediate_upgrades
            )
        if planned_upgrades:
            upgrade_timeline[(current_date + timedelta(months=6)).strftime("%Y-%m")] = (
                planned_upgrades
            )
        if future_upgrades:
            upgrade_timeline[
                (current_date + timedelta(months=18)).strftime("%Y-%m")
            ] = future_upgrades

        # Calculate expected ROI
        total_investment = sum(allocations.values())
        weighted_roi = 0.0
        if total_investment > 0:
            for rec in sorted_recommendations:
                if rec.component_type in allocations:
                    weight = allocations[rec.component_type] / total_investment
                    component_roi = rec.performance_gain / (rec.roi_months or 24)
                    weighted_roi += weight * component_roi

        # Generate cost optimization notes
        optimization_notes = []
        if remaining_budget > budget * 0.1:
            optimization_notes.append(
                f"${remaining_budget:.0f} remaining budget available for additional upgrades"
            )

        if len(immediate_upgrades) > 2:
            optimization_notes.append(
                "Consider spreading upgrades over 6 months to reduce financial impact"
            )

        return BudgetAllocation(
            total_budget=budget,
            allocations=allocations,
            upgrade_timeline=upgrade_timeline,
            cost_optimization_notes=optimization_notes,
            expected_roi=weighted_roi,
        )

    async def scan_market_prices(self):
        """Scan market prices for all tracked components"""
        logger.info("Starting market price scan")

        try:
            # Define product queries for scanning
            product_queries = [
                "AMD Ryzen 9 7900X",
                "Intel Core i9-13900K",
                "NVIDIA RTX 4070 Ti",
                "NVIDIA RTX 4080",
                "G.Skill Trident Z5 64GB DDR5",
                "Samsung 980 PRO 2TB",
            ]

            # Fetch prices from vendors
            products = await self._fetch_product_prices(product_queries)

            # Store in database
            self._store_price_data(products)

            logger.info(f"Market scan completed: {len(products)} products updated")

        except Exception as e:
            logger.error(f"Error during market scan: {e}")

    async def analyze_markets(self):
        """Perform comprehensive market analysis"""
        logger.info("Starting market analysis")

        try:
            # Analyze each component category
            for category in ComponentCategory:
                analysis = self._analyze_market_trends(category)

                # Store analysis in database
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO market_analysis 
                    (category, timestamp, average_price, price_trend, trend_strength, volatility, analysis_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        category.value,
                        analysis.timestamp,
                        analysis.average_price,
                        analysis.price_trend.value,
                        analysis.trend_strength,
                        analysis.volatility,
                        json.dumps(
                            {
                                "best_value_products": analysis.best_value_products,
                                "price_predictions": analysis.price_predictions,
                                "market_insights": analysis.market_insights,
                            }
                        ),
                    ),
                )
                conn.commit()
                conn.close()

                logger.info(
                    f"Analyzed {category.value}: {analysis.price_trend.value} trend, "
                    f"${analysis.average_price:.0f} avg price"
                )

        except Exception as e:
            logger.error(f"Error during market analysis: {e}")

    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate comprehensive upgrade recommendations and budget analysis"""
        logger.info("Generating upgrade recommendations")

        try:
            # Generate upgrade recommendations
            recommendations = self._generate_upgrade_recommendations()

            # Optimize budget allocation
            budget = self.config["budget"]["default_budget"]
            budget_allocation = self._optimize_budget_allocation(
                budget, recommendations
            )

            # Compile comprehensive report
            report = {
                "timestamp": datetime.now().isoformat(),
                "current_system": self.current_system,
                "market_analysis": {
                    cat.value: asdict(analysis)
                    for cat, analysis in self.market_analysis_cache.items()
                },
                "upgrade_recommendations": [asdict(rec) for rec in recommendations],
                "budget_allocation": asdict(budget_allocation),
                "summary": {
                    "total_recommendations": len(recommendations),
                    "critical_upgrades": len(
                        [
                            r
                            for r in recommendations
                            if r.priority == UpgradePriority.CRITICAL
                        ]
                    ),
                    "estimated_total_cost": sum(
                        rec.estimated_cost for rec in recommendations
                    ),
                    "expected_performance_gain": (
                        sum(rec.performance_gain for rec in recommendations)
                        / len(recommendations)
                        if recommendations
                        else 0
                    ),
                },
            }

            logger.info(f"Generated {len(recommendations)} upgrade recommendations")
            return report

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {"error": str(e)}

    def get_price_alerts(self) -> List[Dict[str, Any]]:
        """Get price alerts for tracked products"""
        alerts = []

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Find significant price changes in last 24 hours
            cursor.execute(
                """
                SELECT DISTINCT p.name, p.category, 
                       ph1.price as current_price, ph2.price as previous_price,
                       ph1.vendor, ph1.timestamp
                FROM products p
                JOIN price_history ph1 ON p.product_id = ph1.product_id
                JOIN price_history ph2 ON p.product_id = ph2.product_id
                WHERE ph1.timestamp >= ? AND ph2.timestamp >= ? AND ph2.timestamp < ?
                ORDER BY ph1.timestamp DESC
            """,
                (
                    (datetime.now() - timedelta(hours=24)).isoformat(),
                    (datetime.now() - timedelta(hours=48)).isoformat(),
                    (datetime.now() - timedelta(hours=24)).isoformat(),
                ),
            )

            price_changes = cursor.fetchall()
            conn.close()

            for row in price_changes:
                name, category, current_price, previous_price, vendor, timestamp = row

                if previous_price > 0:
                    price_change = (current_price - previous_price) / previous_price

                    # Check for significant price drops or spikes
                    if (
                        abs(price_change)
                        >= self.config["scanning"]["price_drop_threshold"]
                    ):
                        alert_type = "price_drop" if price_change < 0 else "price_spike"

                        alerts.append(
                            {
                                "type": alert_type,
                                "product": name,
                                "category": category,
                                "vendor": vendor,
                                "current_price": current_price,
                                "previous_price": previous_price,
                                "change_percent": price_change * 100,
                                "timestamp": timestamp,
                            }
                        )

        except Exception as e:
            logger.error(f"Error getting price alerts: {e}")

        return alerts

    def get_status(self) -> Dict[str, Any]:
        """Get current economic scanner status"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Count products and price records
            cursor.execute("SELECT COUNT(*) FROM products")
            product_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM price_history")
            price_record_count = cursor.fetchone()[0]

            # Get last scan time
            cursor.execute("SELECT MAX(timestamp) FROM price_history")
            last_scan = cursor.fetchone()[0]

            conn.close()

            return {
                "running": self.running,
                "products_tracked": product_count,
                "price_records": price_record_count,
                "last_scan": last_scan,
                "market_analyses": len(self.market_analysis_cache),
                "price_cache_size": len(self.price_cache),
                "configuration": {
                    "scan_interval_hours": self.config["scanning"][
                        "price_check_interval_hours"
                    ],
                    "enabled_vendors": len(self.config["vendors"]["enabled_vendors"]),
                    "budget": self.config["budget"]["default_budget"],
                },
            }

        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}


# Global economic scanner instance
economic_scanner = EconomicScanner()

if __name__ == "__main__":
    # CLI interface
    import argparse

    parser = argparse.ArgumentParser(
        description="System Autonomy Core - Economic Intelligence"
    )
    parser.add_argument("--scan", action="store_true", help="Scan market prices")
    parser.add_argument("--analyze", action="store_true", help="Analyze market trends")
    parser.add_argument(
        "--recommend", action="store_true", help="Generate upgrade recommendations"
    )
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--alerts", action="store_true", help="Show price alerts")
    parser.add_argument("--budget", type=float, help="Set budget for analysis")

    args = parser.parse_args()

    async def main():
        if args.budget:
            economic_scanner.config["budget"]["default_budget"] = args.budget
            economic_scanner._save_config(economic_scanner.config)
            print(f"💰 Budget set to ${args.budget:.0f}")

        if args.scan:
            print("📊 Scanning market prices...")
            await economic_scanner.scan_market_prices()
            print("✅ Market scan completed")

        if args.analyze:
            print("📈 Analyzing market trends...")
            await economic_scanner.analyze_markets()
            print("✅ Market analysis completed")

        if args.recommend:
            print("🎯 Generating upgrade recommendations...")
            report = economic_scanner.generate_recommendations()

            if "error" not in report:
                print(f"\n💡 Upgrade Recommendations:")
                for rec in report["upgrade_recommendations"]:
                    print(
                        f"   {rec['component_type'].upper()}: {rec['recommended_component']}"
                    )
                    print(f"      Priority: {rec['priority'].upper()}")
                    print(f"      Cost: ${rec['estimated_cost']:.0f}")
                    print(f"      Performance Gain: {rec['performance_gain']:.1f}%")
                    print(f"      Best Timing: {rec['best_timing']}")
                    print()

                budget_alloc = report["budget_allocation"]
                print(
                    f"💰 Budget Allocation (${budget_alloc['total_budget']:.0f} total):"
                )
                for category, amount in budget_alloc["allocations"].items():
                    print(f"   {category.upper()}: ${amount:.0f}")
            else:
                print(f"❌ Error: {report['error']}")

        if args.alerts:
            alerts = economic_scanner.get_price_alerts()
            if alerts:
                print("🚨 Price Alerts:")
                for alert in alerts[:10]:  # Show last 10 alerts
                    change_symbol = "📉" if alert["type"] == "price_drop" else "📈"
                    print(
                        f"   {change_symbol} {alert['product']}: {alert['change_percent']:+.1f}% "
                        f"(${alert['current_price']:.0f}) at {alert['vendor']}"
                    )
            else:
                print("✅ No significant price changes")

        if args.status:
            status = economic_scanner.get_status()
            if "error" not in status:
                print("📊 Economic Scanner Status:")
                print(f"   Products Tracked: {status['products_tracked']}")
                print(f"   Price Records: {status['price_records']}")
                print(f"   Last Scan: {status['last_scan']}")
                print(f"   Market Analyses: {status['market_analyses']}")
                print(f"   Budget: ${status['configuration']['budget']:.0f}")
            else:
                print(f"❌ Error: {status['error']}")

        if not any(
            [
                args.scan,
                args.analyze,
                args.recommend,
                args.alerts,
                args.status,
                args.budget,
            ]
        ):
            # Default: show quick status
            status = economic_scanner.get_status()
            if "error" not in status:
                print("💰 Economic Intelligence Quick Status:")
                print(f"   Products: {status['products_tracked']}")
                print(f"   Last Scan: {status['last_scan'] or 'Never'}")
                print(f"   Budget: ${status['configuration']['budget']:.0f}")

            # Show recent alerts
            alerts = economic_scanner.get_price_alerts()
            if alerts:
                print(f"   Recent Alerts: {len(alerts)}")

    # Run async main
    asyncio.run(main())
