"""
Financial Monitoring Engine
===========================

Privacy-conscious financial behavior analysis with bank CSV parsing,
expense categorization, spending pattern analysis, and budget prediction.
"""

import os
import csv
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re
from collections import defaultdict
import pandas as pd
import numpy as np

from ..database.user_profile_schema import UserProfileDatabase, FinancialStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinancialConfig:
    """Configuration for financial monitoring"""

    csv_directory: str = "input_data/financial"
    auto_categorization: bool = True
    privacy_mode: bool = True  # Abstract amounts rather than store exact values
    trend_analysis_days: int = 90
    budget_prediction_months: int = 3
    stress_threshold: float = 0.8  # Spending threshold for stress calculation
    enable_plaid_integration: bool = False  # Requires API keys
    alert_on_unusual_spending: bool = True

    def __post_init__(self):
        # Load Plaid configuration from environment variables
        plaid_client_id = os.getenv("PLAID_CLIENT_ID")
        plaid_secret = os.getenv("PLAID_SECRET")

        # Enable Plaid integration if credentials are available
        if plaid_client_id and plaid_secret:
            self.enable_plaid_integration = True

        # Load other environment-based configurations
        csv_dir = os.getenv("FINANCIAL_CSV_DIRECTORY")
        if csv_dir:
            self.csv_directory = csv_dir

        privacy_mode = os.getenv("FINANCIAL_PRIVACY_MODE")
        if privacy_mode:
            self.privacy_mode = privacy_mode.lower() in ("true", "1", "yes")

        trend_days = os.getenv("FINANCIAL_TREND_ANALYSIS_DAYS")
        if trend_days:
            try:
                self.trend_analysis_days = int(trend_days)
            except ValueError:
                pass


@dataclass
class Transaction:
    """Standardized transaction structure"""

    id: str
    date: datetime
    description: str
    amount: float
    category: str = ""
    subcategory: str = ""
    account: str = ""
    transaction_type: str = ""  # debit, credit, transfer
    merchant: str = ""
    is_recurring: bool = False
    confidence_score: float = 0.0


class ExpenseCategorizer:
    """Intelligent expense categorization engine"""

    def __init__(self):
        self.category_keywords = {
            "groceries": [
                "grocery",
                "supermarket",
                "walmart",
                "target",
                "kroger",
                "safeway",
                "whole foods",
                "costco",
                "sams club",
                "food",
                "market",
            ],
            "dining": [
                "restaurant",
                "cafe",
                "coffee",
                "starbucks",
                "mcdonald",
                "pizza",
                "burger",
                "taco",
                "doordash",
                "uber eats",
                "grubhub",
                "delivery",
            ],
            "transportation": [
                "gas",
                "fuel",
                "uber",
                "lyft",
                "taxi",
                "bus",
                "metro",
                "parking",
                "car wash",
                "auto",
                "repair",
                "maintenance",
                "insurance",
            ],
            "entertainment": [
                "movie",
                "theater",
                "netflix",
                "spotify",
                "gaming",
                "steam",
                "concert",
                "event",
                "ticket",
                "entertainment",
                "subscription",
            ],
            "shopping": [
                "amazon",
                "ebay",
                "clothing",
                "apparel",
                "electronics",
                "best buy",
                "apple store",
                "retail",
                "department store",
                "online",
            ],
            "healthcare": [
                "medical",
                "doctor",
                "pharmacy",
                "hospital",
                "dental",
                "health",
                "prescription",
                "clinic",
                "insurance",
                "copay",
            ],
            "education": [
                "school",
                "university",
                "tuition",
                "textbook",
                "student",
                "course",
                "training",
                "education",
                "library",
            ],
            "utilities": [
                "electric",
                "gas bill",
                "water",
                "internet",
                "phone",
                "cable",
                "utility",
                "bill",
                "service",
            ],
            "housing": [
                "rent",
                "mortgage",
                "property",
                "home",
                "apartment",
                "housing",
                "maintenance",
                "repair",
                "furniture",
            ],
            "income": [
                "salary",
                "paycheck",
                "deposit",
                "bonus",
                "refund",
                "transfer in",
                "income",
                "wages",
                "payment received",
            ],
            "investment": [
                "investment",
                "stock",
                "dividend",
                "mutual fund",
                "retirement",
                "401k",
                "ira",
                "savings",
                "broker",
            ],
            "debt": [
                "loan",
                "credit card",
                "interest",
                "fee",
                "overdraft",
                "late fee",
                "penalty",
                "debt",
            ],
        }

        # Merchant-specific mappings
        self.merchant_categories = {
            "amazon": "shopping",
            "walmart": "groceries",
            "target": "shopping",
            "starbucks": "dining",
            "netflix": "entertainment",
            "spotify": "entertainment",
            "uber": "transportation",
            "lyft": "transportation",
        }

    def categorize_transaction(
        self, transaction: Transaction
    ) -> Tuple[str, str, float]:
        """Categorize transaction and return category, subcategory, confidence"""
        description = transaction.description.lower()

        # Check merchant-specific categories first
        for merchant, category in self.merchant_categories.items():
            if merchant in description:
                return category, merchant, 0.9

        # Check keyword-based categories
        best_category = "other"
        best_subcategory = ""
        best_confidence = 0.0

        for category, keywords in self.category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in description)
            confidence = matches / len(keywords) if keywords else 0

            if confidence > best_confidence:
                best_confidence = confidence
                best_category = category
                # Try to extract more specific subcategory
                best_subcategory = self._extract_subcategory(description, category)

        # Special handling for amounts (income vs expense)
        if transaction.amount > 0 and best_category == "other":
            if any(
                keyword in description for keyword in ["deposit", "transfer", "payroll"]
            ):
                best_category = "income"
                best_confidence = 0.7

        return best_category, best_subcategory, best_confidence

    def _extract_subcategory(self, description: str, category: str) -> str:
        """Extract more specific subcategory"""
        if category == "dining":
            if any(word in description for word in ["coffee", "starbucks"]):
                return "coffee"
            elif any(word in description for word in ["pizza", "burger", "fast"]):
                return "fast_food"
            else:
                return "restaurant"

        elif category == "transportation":
            if any(word in description for word in ["gas", "fuel"]):
                return "fuel"
            elif any(word in description for word in ["uber", "lyft", "taxi"]):
                return "rideshare"
            else:
                return "other_transport"

        elif category == "shopping":
            if "amazon" in description:
                return "online"
            elif any(word in description for word in ["clothing", "apparel"]):
                return "clothing"
            else:
                return "general"

        return ""


class FinancialAnalyzer:
    """Advanced financial behavior analysis"""

    def __init__(self, config: FinancialConfig):
        self.config = config

    def abstract_amount(self, amount: float) -> str:
        """Convert exact amount to privacy-conscious tier"""
        abs_amount = abs(amount)

        if abs_amount < 10:
            return "very_low"
        elif abs_amount < 50:
            return "low"
        elif abs_amount < 200:
            return "medium"
        elif abs_amount < 1000:
            return "high"
        else:
            return "very_high"

    def analyze_spending_patterns(
        self, transactions: List[Transaction]
    ) -> Dict[str, Any]:
        """Analyze spending patterns and behaviors"""
        if not transactions:
            return {}

        # Convert to DataFrame for easier analysis
        data = []
        for t in transactions:
            data.append(
                {
                    "date": t.date,
                    "amount": t.amount,
                    "category": t.category,
                    "subcategory": t.subcategory,
                    "description": t.description,
                }
            )

        df = pd.DataFrame(data)

        analysis = {
            "total_transactions": len(transactions),
            "date_range": {
                "start": df["date"].min().isoformat() if not df.empty else None,
                "end": df["date"].max().isoformat() if not df.empty else None,
            },
            "spending_by_category": {},
            "income_vs_expense": {},
            "monthly_trends": {},
            "weekly_patterns": {},
            "unusual_transactions": [],
            "recurring_patterns": {},
            "financial_stress_indicators": {},
        }

        # Spending by category
        expenses = df[df["amount"] < 0]  # Negative amounts are expenses
        if not expenses.empty:
            category_spending = expenses.groupby("category")["amount"].sum().abs()
            analysis["spending_by_category"] = category_spending.to_dict()

        # Income vs expense analysis
        income = df[df["amount"] > 0]["amount"].sum()
        expense = abs(df[df["amount"] < 0]["amount"].sum())

        analysis["income_vs_expense"] = {
            "total_income": (
                self.abstract_amount(income) if self.config.privacy_mode else income
            ),
            "total_expenses": (
                self.abstract_amount(expense) if self.config.privacy_mode else expense
            ),
            "net_flow": (
                self.abstract_amount(income - expense)
                if self.config.privacy_mode
                else (income - expense)
            ),
            "expense_ratio": expense / income if income > 0 else 0,
        }

        # Monthly trends
        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month")["amount"].sum()
        analysis["monthly_trends"] = {str(k): v for k, v in monthly.to_dict().items()}

        # Weekly patterns (day of week spending)
        df["day_of_week"] = df["date"].dt.day_name()
        weekly = df[df["amount"] < 0].groupby("day_of_week")["amount"].sum().abs()
        analysis["weekly_patterns"] = weekly.to_dict()

        # Identify unusual transactions (outliers)
        if not expenses.empty:
            expense_amounts = expenses["amount"].abs()
            q75, q25 = np.percentile(expense_amounts, [75, 25])
            iqr = q75 - q25
            outlier_threshold = q75 + (1.5 * iqr)

            outliers = expenses[expense_amounts > outlier_threshold]
            analysis["unusual_transactions"] = [
                {
                    "date": row["date"].isoformat(),
                    "amount_tier": self.abstract_amount(abs(row["amount"])),
                    "description": row["description"][:50],
                    "category": row["category"],
                }
                for _, row in outliers.head(10).iterrows()
            ]

        # Financial stress indicators
        analysis["financial_stress_indicators"] = self._calculate_stress_indicators(df)

        return analysis

    def _calculate_stress_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various financial stress indicators"""
        stress_indicators = {}

        if df.empty:
            return stress_indicators

        # Income volatility
        income_transactions = df[df["amount"] > 0]
        if len(income_transactions) > 1:
            income_std = income_transactions["amount"].std()
            income_mean = income_transactions["amount"].mean()
            income_volatility = income_std / income_mean if income_mean > 0 else 0
            stress_indicators["income_volatility"] = min(income_volatility, 1.0)

        # Expense ratio (expenses / income)
        total_income = df[df["amount"] > 0]["amount"].sum()
        total_expenses = abs(df[df["amount"] < 0]["amount"].sum())

        if total_income > 0:
            expense_ratio = total_expenses / total_income
            stress_indicators["expense_ratio"] = min(expense_ratio, 1.0)

        # Frequency of overdrafts or fees
        fee_transactions = df[
            df["description"].str.contains(
                "fee|overdraft|penalty", case=False, na=False
            )
        ]
        if not fee_transactions.empty:
            fee_frequency = len(fee_transactions) / len(df)
            stress_indicators["fee_frequency"] = fee_frequency

        # Debt payment ratio
        debt_payments = df[df["category"] == "debt"]
        if not debt_payments.empty and total_income > 0:
            debt_ratio = abs(debt_payments["amount"].sum()) / total_income
            stress_indicators["debt_ratio"] = min(debt_ratio, 1.0)

        return stress_indicators

    def predict_budget(
        self, transactions: List[Transaction], months_ahead: int = 3
    ) -> Dict[str, Any]:
        """Predict future budget based on historical patterns"""
        if not transactions:
            return {}

        # Convert to DataFrame
        data = [
            {"date": t.date, "amount": t.amount, "category": t.category}
            for t in transactions
        ]

        df = pd.DataFrame(data)

        # Calculate monthly averages by category
        df["month"] = df["date"].dt.to_period("M")
        monthly_by_category = (
            df.groupby(["month", "category"])["amount"].sum().unstack(fill_value=0)
        )

        predictions = {}

        for category in monthly_by_category.columns:
            category_data = monthly_by_category[category]

            # Simple trend analysis (could be enhanced with more sophisticated models)
            if len(category_data) >= 2:
                # Calculate trend
                x = np.arange(len(category_data))
                y = category_data.values

                # Simple linear regression
                if len(x) > 1:
                    slope = np.corrcoef(x, y)[0, 1] * np.std(y) / np.std(x)
                    intercept = np.mean(y) - slope * np.mean(x)

                    # Predict future months
                    future_predictions = []
                    for i in range(1, months_ahead + 1):
                        future_x = len(category_data) + i
                        predicted_amount = slope * future_x + intercept

                        future_predictions.append(
                            {
                                "month_offset": i,
                                "predicted_amount_tier": (
                                    self.abstract_amount(abs(predicted_amount))
                                    if self.config.privacy_mode
                                    else predicted_amount
                                ),
                                "trend": (
                                    "increasing"
                                    if slope > 0
                                    else "decreasing" if slope < 0 else "stable"
                                ),
                            }
                        )

                    predictions[category] = {
                        "historical_average": (
                            self.abstract_amount(abs(category_data.mean()))
                            if self.config.privacy_mode
                            else category_data.mean()
                        ),
                        "trend_slope": slope,
                        "predictions": future_predictions,
                    }

        return predictions


class CSVParser:
    """Parse various bank CSV formats"""

    def __init__(self):
        # Common CSV field mappings for different banks
        self.field_mappings = {
            "chase": {
                "date": ["Transaction Date", "Date"],
                "description": ["Description"],
                "amount": ["Amount"],
                "type": ["Type"],
                "balance": ["Balance"],
            },
            "bank_of_america": {
                "date": ["Date"],
                "description": ["Description"],
                "amount": ["Amount"],
                "running_balance": ["Running Bal."],
            },
            "wells_fargo": {
                "date": ["Date"],
                "amount": ["Amount"],
                "description": ["Description"],
            },
            "generic": {
                "date": ["date", "transaction_date", "Date"],
                "description": ["description", "memo", "Description"],
                "amount": ["amount", "Amount"],
                "category": ["category", "Category"],
            },
        }

    def detect_format(self, csv_path: str) -> str:
        """Detect bank CSV format"""
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames

                # Check against known formats
                for bank, mapping in self.field_mappings.items():
                    if bank == "generic":
                        continue

                    required_fields = list(mapping.keys())
                    matches = 0

                    for field in required_fields:
                        field_options = mapping[field]
                        if any(option in headers for option in field_options):
                            matches += 1

                    if matches >= len(required_fields) * 0.7:  # 70% match threshold
                        return bank

                return "generic"

        except Exception as e:
            logger.error(f"Error detecting CSV format: {e}")
            return "generic"

    def parse_csv(self, csv_path: str, bank_format: str = None) -> List[Transaction]:
        """Parse bank CSV file into standardized transactions"""
        if bank_format is None:
            bank_format = self.detect_format(csv_path)

        if bank_format not in self.field_mappings:
            bank_format = "generic"

        mapping = self.field_mappings[bank_format]
        transactions = []

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Extract fields using mapping
                    date_str = self._get_field_value(row, mapping.get("date", []))
                    description = self._get_field_value(
                        row, mapping.get("description", [])
                    )
                    amount_str = self._get_field_value(row, mapping.get("amount", []))

                    if not all([date_str, description, amount_str]):
                        continue

                    # Parse date
                    try:
                        date = self._parse_date(date_str)
                    except:
                        continue

                    # Parse amount
                    try:
                        amount = self._parse_amount(amount_str)
                    except:
                        continue

                    # Create transaction
                    transaction = Transaction(
                        id=hashlib.md5(
                            f"{date_str}{description}{amount_str}".encode()
                        ).hexdigest()[:12],
                        date=date,
                        description=description.strip(),
                        amount=amount,
                        account=os.path.basename(csv_path),
                    )

                    transactions.append(transaction)

        except Exception as e:
            logger.error(f"Error parsing CSV {csv_path}: {e}")

        return transactions

    def _get_field_value(self, row: Dict, field_options: List[str]) -> str:
        """Get field value from row using multiple possible field names"""
        for field in field_options:
            if field in row and row[field]:
                return row[field]
        return ""

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string in various formats"""
        date_formats = [
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%Y-%m-%d",
            "%m/%d/%y",
            "%m-%d-%y",
            "%y-%m-%d",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d/%m/%Y",
            "%d-%m-%Y",
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse date: {date_str}")

    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string, handling various formats"""
        # Remove currency symbols and spaces
        amount_str = re.sub(r"[$,\s]", "", amount_str.strip())

        # Handle parentheses for negative amounts
        if amount_str.startswith("(") and amount_str.endswith(")"):
            amount_str = "-" + amount_str[1:-1]

        return float(amount_str)


class PlaidIntegration:
    """Plaid API integration for real-time bank account data"""

    def __init__(self, config: FinancialConfig):
        self.config = config
        self.client_id = os.getenv("PLAID_CLIENT_ID", "")
        self.secret = os.getenv("PLAID_SECRET", "")
        self.environment = os.getenv(
            "PLAID_ENVIRONMENT", "sandbox"
        )  # sandbox, development, production
        self.access_tokens = []  # Store user access tokens

    def initialize_client(self):
        """Initialize Plaid client"""
        try:
            from plaid.api import plaid_api
            from plaid.model.transactions_get_request import TransactionsGetRequest
            from plaid.model.accounts_get_request import AccountsGetRequest
            from plaid.configuration import Configuration
            from plaid.api_client import ApiClient

            # Map environment strings to Plaid configuration
            env_map = {
                "sandbox": Configuration.environment.sandbox,
                "development": Configuration.environment.development,
                "production": Configuration.environment.production,
            }

            configuration = Configuration(
                host=env_map.get(self.environment, Configuration.environment.sandbox),
                api_key={"clientId": self.client_id, "secret": self.secret},
            )

            api_client = ApiClient(configuration)
            self.client = plaid_api.PlaidApi(api_client)
            return True

        except ImportError:
            logger.error(
                "Plaid SDK not available. Install with: pip install plaid-python"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Plaid client: {e}")
            return False

    def fetch_transactions(
        self, access_token: str, start_date: datetime = None, end_date: datetime = None
    ) -> List[Transaction]:
        """Fetch transactions from Plaid API"""
        if not hasattr(self, "client"):
            if not self.initialize_client():
                return []

        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        try:
            from plaid.model.transactions_get_request import TransactionsGetRequest
            from datetime import date

            request = TransactionsGetRequest(
                access_token=access_token,
                start_date=start_date.date(),
                end_date=end_date.date(),
            )

            response = self.client.transactions_get(request)
            transactions = []

            for plaid_transaction in response["transactions"]:
                # Convert Plaid transaction to our Transaction format
                transaction = Transaction(
                    id=plaid_transaction["transaction_id"],
                    date=datetime.strptime(plaid_transaction["date"], "%Y-%m-%d"),
                    description=plaid_transaction["name"],
                    amount=plaid_transaction[
                        "amount"
                    ],  # Plaid amounts are positive for outflows
                    category=(
                        plaid_transaction.get("category", [""])[0]
                        if plaid_transaction.get("category")
                        else ""
                    ),
                    subcategory=(
                        plaid_transaction.get("category", ["", ""])[1]
                        if len(plaid_transaction.get("category", [])) > 1
                        else ""
                    ),
                    account=plaid_transaction["account_id"],
                    transaction_type=(
                        "debit" if plaid_transaction["amount"] > 0 else "credit"
                    ),
                    merchant=plaid_transaction.get("merchant_name", ""),
                    confidence_score=1.0,  # Plaid data is authoritative
                )

                transactions.append(transaction)

            logger.info(f"Fetched {len(transactions)} transactions from Plaid")
            return transactions

        except Exception as e:
            logger.error(f"Error fetching Plaid transactions: {e}")
            return []

    def fetch_accounts(self, access_token: str) -> List[Dict[str, Any]]:
        """Fetch account information from Plaid"""
        if not hasattr(self, "client"):
            if not self.initialize_client():
                return []

        try:
            from plaid.model.accounts_get_request import AccountsGetRequest

            request = AccountsGetRequest(access_token=access_token)
            response = self.client.accounts_get(request)

            accounts = []
            for account in response["accounts"]:
                account_info = {
                    "account_id": account["account_id"],
                    "name": account["name"],
                    "type": account["type"],
                    "subtype": account["subtype"],
                    "balance": account["balances"]["current"],
                    "available_balance": account["balances"].get("available"),
                    "currency": account["balances"].get("iso_currency_code", "USD"),
                }
                accounts.append(account_info)

            return accounts

        except Exception as e:
            logger.error(f"Error fetching Plaid accounts: {e}")
            return []

    def create_link_token(self, user_id: str) -> str:
        """Create a link token for Plaid Link initialization"""
        if not hasattr(self, "client"):
            if not self.initialize_client():
                return ""

        try:
            from plaid.model.link_token_create_request import LinkTokenCreateRequest
            from plaid.model.link_token_create_request_user import (
                LinkTokenCreateRequestUser,
            )
            from plaid.model.country_code import CountryCode
            from plaid.model.products import Products

            request = LinkTokenCreateRequest(
                products=[Products("transactions")],
                client_name="Vega2.0 Financial Monitor",
                country_codes=[CountryCode("US")],
                language="en",
                user=LinkTokenCreateRequestUser(client_user_id=user_id),
            )

            response = self.client.link_token_create(request)
            return response["link_token"]

        except Exception as e:
            logger.error(f"Error creating Plaid link token: {e}")
            return ""


class FinanceMonitor:
    """Main financial monitoring engine"""

    def __init__(self, db: UserProfileDatabase, config: FinancialConfig = None):
        self.db = db
        self.config = config or FinancialConfig()
        self.categorizer = ExpenseCategorizer()
        self.analyzer = FinancialAnalyzer(self.config)
        self.csv_parser = CSVParser()

        # Initialize Plaid integration if enabled
        self.plaid_integration = None
        if self.config.enable_plaid_integration:
            self.plaid_integration = PlaidIntegration(self.config)

    async def process_csv_files(self, directory: str = None) -> Dict[str, Any]:
        """Process all CSV files in directory"""
        if directory is None:
            directory = self.config.csv_directory

        directory_path = Path(directory)
        if not directory_path.exists():
            logger.warning(f"CSV directory {directory} does not exist")
            return {}

        results = {
            "processed_files": [],
            "total_transactions": 0,
            "categorized_transactions": 0,
            "analysis": {},
            "errors": [],
        }

        all_transactions = []

        # Process each CSV file
        for csv_file in directory_path.glob("*.csv"):
            try:
                logger.info(f"Processing {csv_file}")
                transactions = self.csv_parser.parse_csv(str(csv_file))

                # Categorize transactions
                for transaction in transactions:
                    category, subcategory, confidence = (
                        self.categorizer.categorize_transaction(transaction)
                    )
                    transaction.category = category
                    transaction.subcategory = subcategory
                    transaction.confidence_score = confidence

                all_transactions.extend(transactions)

                results["processed_files"].append(
                    {"filename": csv_file.name, "transactions": len(transactions)}
                )

            except Exception as e:
                error_msg = f"Error processing {csv_file}: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)

        results["total_transactions"] = len(all_transactions)
        results["categorized_transactions"] = sum(
            1 for t in all_transactions if t.category != "other"
        )

        # Store transactions
        if all_transactions:
            await self._store_transactions(all_transactions)

        # Run analysis
        if all_transactions:
            results["analysis"] = self.analyzer.analyze_spending_patterns(
                all_transactions
            )
            results["budget_predictions"] = self.analyzer.predict_budget(
                all_transactions, self.config.budget_prediction_months
            )

        return results

    async def process_plaid_accounts(
        self, access_tokens: List[str] = None
    ) -> Dict[str, Any]:
        """Process transactions from connected Plaid accounts"""
        if not self.plaid_integration:
            return {"error": "Plaid integration not enabled"}

        if not access_tokens:
            access_tokens = getattr(self, "stored_access_tokens", [])

        results = {
            "processed_accounts": [],
            "total_transactions": 0,
            "categorized_transactions": 0,
            "errors": [],
            "analysis": {},
            "budget_predictions": {},
        }

        all_transactions = []

        for access_token in access_tokens:
            try:
                # Fetch account information
                accounts = self.plaid_integration.fetch_accounts(access_token)

                # Fetch transactions for the past 30 days
                transactions = self.plaid_integration.fetch_transactions(
                    access_token,
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now(),
                )

                # Categorize transactions
                for transaction in transactions:
                    category, confidence = self.categorizer.categorize_transaction(
                        transaction
                    )
                    transaction.category = category
                    transaction.confidence_score = confidence

                all_transactions.extend(transactions)

                results["processed_accounts"].append(
                    {
                        "access_token": access_token[:10]
                        + "...",  # Partial token for logging
                        "accounts": len(accounts),
                        "transactions": len(transactions),
                        "account_types": [acc["type"] for acc in accounts],
                    }
                )

            except Exception as e:
                error_msg = (
                    f"Error processing Plaid account {access_token[:10]}...: {e}"
                )
                results["errors"].append(error_msg)
                logger.error(error_msg)

        results["total_transactions"] = len(all_transactions)
        results["categorized_transactions"] = sum(
            1 for t in all_transactions if t.category != "other"
        )

        # Store transactions
        if all_transactions:
            await self._store_transactions(all_transactions)

        # Run analysis
        if all_transactions:
            results["analysis"] = self.analyzer.analyze_spending_patterns(
                all_transactions
            )
            results["budget_predictions"] = self.analyzer.predict_budget(
                all_transactions, self.config.budget_prediction_months
            )

        return results

    async def _store_transactions(self, transactions: List[Transaction]):
        """Store transactions in database with privacy abstraction"""
        session = self.db.get_session()

        try:
            for transaction in transactions:
                # Check if transaction already exists
                existing = (
                    session.query(FinancialStatus)
                    .filter(
                        FinancialStatus.category == "transaction",
                        FinancialStatus.data_source == f"csv_{transaction.account}",
                        FinancialStatus.created_at.cast(datetime).date()
                        == transaction.date.date(),
                    )
                    .first()
                )

                if not existing:
                    # Determine transaction type and trends
                    if transaction.amount > 0:
                        trans_category = "income"
                        amount_tier = self.analyzer.abstract_amount(transaction.amount)
                    else:
                        trans_category = "expense"
                        amount_tier = self.analyzer.abstract_amount(
                            abs(transaction.amount)
                        )

                    financial_record = FinancialStatus(
                        category=trans_category,
                        subcategory=transaction.category,
                        amount_tier=amount_tier,
                        frequency="one_time",
                        trend="stable",  # Could be enhanced with historical analysis
                        spending_pattern=transaction.subcategory,
                        financial_stress_indicator=0.0,  # Calculated separately
                        confidence_score=transaction.confidence_score,
                        data_source=f"csv_{transaction.account}",
                        created_at=transaction.date,
                    )

                    session.add(financial_record)

            session.commit()
            logger.info(f"Stored {len(transactions)} financial records")

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing financial data: {e}")
        finally:
            session.close()

    def get_financial_summary(self) -> Dict[str, Any]:
        """Get current financial summary"""
        session = self.db.get_session()

        try:
            # Get recent financial data
            recent_date = datetime.now() - timedelta(days=30)

            records = (
                session.query(FinancialStatus)
                .filter(
                    FinancialStatus.created_at >= recent_date,
                    FinancialStatus.is_active == True,
                )
                .all()
            )

            summary = {
                "total_records": len(records),
                "categories": {},
                "spending_tiers": {},
                "stress_indicators": [],
                "recent_trends": {},
            }

            # Categorize records
            for record in records:
                category = record.subcategory or record.category
                summary["categories"][category] = (
                    summary["categories"].get(category, 0) + 1
                )

                tier = record.amount_tier
                summary["spending_tiers"][tier] = (
                    summary["spending_tiers"].get(tier, 0) + 1
                )

                if record.financial_stress_indicator > 0.5:
                    summary["stress_indicators"].append(
                        {
                            "category": category,
                            "stress_level": record.financial_stress_indicator,
                            "date": record.created_at.isoformat(),
                        }
                    )

            return summary

        finally:
            session.close()


async def run_financial_monitoring(
    db_path: str = None, config_dict: Dict = None, access_tokens: List[str] = None
) -> Dict[str, Any]:
    """Main function to run financial monitoring"""
    db = UserProfileDatabase(db_path)

    config = FinancialConfig()
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    monitor = FinanceMonitor(db, config)

    # Collect results from all data sources
    all_results = {}

    # Process CSV files
    csv_results = await monitor.process_csv_files()
    all_results["csv_processing"] = csv_results

    # Process Plaid accounts if enabled and tokens provided
    if config.enable_plaid_integration and access_tokens:
        plaid_results = await monitor.process_plaid_accounts(access_tokens)
        all_results["plaid_processing"] = plaid_results

    # Calculate summary statistics
    total_transactions = 0
    total_transactions += csv_results.get("total_transactions", 0)
    if "plaid_processing" in all_results:
        total_transactions += all_results["plaid_processing"].get(
            "total_transactions", 0
        )

    all_results["summary"] = {
        "total_transactions_processed": total_transactions,
        "data_sources": list(all_results.keys()),
        "processing_timestamp": datetime.now().isoformat(),
    }

    return all_results


if __name__ == "__main__":
    # Test financial monitoring
    async def main():
        results = await run_financial_monitoring()
        print("Financial Monitoring Results:")
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())
