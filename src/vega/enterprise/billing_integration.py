"""
Enterprise Billing Integration System
===================================

Comprehensive billing and subscription management with support for
multiple payment providers and automated billing workflows.

Features:
- Multi-provider billing integration (Stripe, Paddle, Chargebee)
- Automated subscription lifecycle management
- Usage-based billing and metering
- Invoice generation and payment processing
- Dunning management and failed payment handling
- Revenue analytics and reporting
"""

import asyncio
import logging
import json
import uuid
import hmac
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from decimal import Decimal
import httpx

logger = logging.getLogger(__name__)


class BillingProvider(Enum):
    """Supported billing providers"""

    STRIPE = "stripe"
    PADDLE = "paddle"
    CHARGEBEE = "chargebee"
    PAYPAL = "paypal"
    CUSTOM = "custom"


class BillingInterval(Enum):
    """Billing interval types"""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    USAGE_BASED = "usage_based"
    ONE_TIME = "one_time"


class PaymentStatus(Enum):
    """Payment status types"""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class InvoiceStatus(Enum):
    """Invoice status types"""

    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    VOID = "void"
    UNCOLLECTIBLE = "uncollectible"


@dataclass
class BillingCustomer:
    """Billing customer information"""

    customer_id: str
    tenant_id: str
    email: str
    name: str
    company: Optional[str] = None

    # Billing details
    billing_address: Dict[str, str] = field(default_factory=dict)
    payment_methods: List[Dict[str, Any]] = field(default_factory=list)
    default_payment_method: Optional[str] = None

    # Provider-specific IDs
    stripe_customer_id: Optional[str] = None
    paddle_customer_id: Optional[str] = None
    chargebee_customer_id: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subscription:
    """Subscription configuration"""

    subscription_id: str
    customer_id: str
    tenant_id: str
    plan_id: str

    # Billing configuration
    billing_interval: BillingInterval
    amount: Decimal
    currency: str = "USD"

    # Status and lifecycle
    status: str = "active"  # active, cancelled, past_due, trialing
    current_period_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    current_period_end: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=30)
    )
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    # Usage tracking
    usage_records: List[Dict[str, Any]] = field(default_factory=list)
    metered_usage: Dict[str, Decimal] = field(default_factory=dict)

    # Provider-specific data
    provider_subscription_id: Optional[str] = None
    provider_data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Invoice:
    """Invoice information"""

    invoice_id: str
    subscription_id: str
    customer_id: str
    tenant_id: str

    # Invoice details
    status: InvoiceStatus
    amount_due: Decimal
    amount_paid: Decimal = Decimal("0")
    currency: str = "USD"

    # Line items
    line_items: List[Dict[str, Any]] = field(default_factory=list)

    # Dates
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=30)
    )
    paid_at: Optional[datetime] = None

    # Provider data
    provider_invoice_id: Optional[str] = None
    provider_data: Dict[str, Any] = field(default_factory=dict)


class BillingManager:
    """Comprehensive billing management system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.http_client = httpx.AsyncClient()

        # Provider configurations
        self.stripe_config = config.get("stripe", {})
        self.paddle_config = config.get("paddle", {})
        self.chargebee_config = config.get("chargebee", {})

        # Data storage
        self.customers: Dict[str, BillingCustomer] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        self.invoices: Dict[str, Invoice] = {}

        # Usage tracking
        self.usage_buffer: List[Dict[str, Any]] = []
        self.usage_aggregation_interval = 3600  # 1 hour

    async def initialize(self):
        """Initialize billing manager"""

        logger.info("Initializing billing manager")

        # Validate provider configurations
        await self._validate_provider_configs()

        # Start background tasks
        asyncio.create_task(self._usage_aggregation_task())
        asyncio.create_task(self._invoice_generation_task())
        asyncio.create_task(self._payment_retry_task())

    async def _validate_provider_configs(self):
        """Validate billing provider configurations"""

        # Validate Stripe config
        if self.stripe_config.get("enabled", False):
            required_keys = ["secret_key", "webhook_secret"]
            for key in required_keys:
                if not self.stripe_config.get(key):
                    logger.warning(f"Stripe {key} not configured")

        # Validate other providers similarly

    async def create_customer(
        self,
        tenant_id: str,
        email: str,
        name: str,
        company: Optional[str] = None,
        billing_address: Optional[Dict[str, str]] = None,
    ) -> BillingCustomer:
        """Create a new billing customer"""

        customer_id = str(uuid.uuid4())

        customer = BillingCustomer(
            customer_id=customer_id,
            tenant_id=tenant_id,
            email=email,
            name=name,
            company=company,
            billing_address=billing_address or {},
        )

        # Create customer in enabled billing providers
        await self._create_customer_in_providers(customer)

        # Store customer
        self.customers[customer_id] = customer

        logger.info(f"Created billing customer: {customer_id} for tenant: {tenant_id}")

        return customer

    async def _create_customer_in_providers(self, customer: BillingCustomer):
        """Create customer in all enabled billing providers"""

        # Create in Stripe
        if self.stripe_config.get("enabled", False):
            stripe_customer = await self._create_stripe_customer(customer)
            customer.stripe_customer_id = stripe_customer.get("id")

        # Create in Paddle
        if self.paddle_config.get("enabled", False):
            paddle_customer = await self._create_paddle_customer(customer)
            customer.paddle_customer_id = paddle_customer.get("id")

        # Create in Chargebee
        if self.chargebee_config.get("enabled", False):
            chargebee_customer = await self._create_chargebee_customer(customer)
            customer.chargebee_customer_id = chargebee_customer.get("id")

    async def _create_stripe_customer(
        self, customer: BillingCustomer
    ) -> Dict[str, Any]:
        """Create customer in Stripe"""

        url = "https://api.stripe.com/v1/customers"
        headers = {
            "Authorization": f"Bearer {self.stripe_config['secret_key']}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {
            "email": customer.email,
            "name": customer.name,
            "metadata[tenant_id]": customer.tenant_id,
            "metadata[customer_id]": customer.customer_id,
        }

        if customer.company:
            data["name"] = f"{customer.name} ({customer.company})"

        response = await self.http_client.post(url, headers=headers, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to create Stripe customer: {response.text}")
            return {}

    async def _create_paddle_customer(
        self, customer: BillingCustomer
    ) -> Dict[str, Any]:
        """Create customer in Paddle"""

        # Paddle API implementation
        # This would use Paddle's customer creation API

        return {"id": f"paddle_{customer.customer_id}"}

    async def _create_chargebee_customer(
        self, customer: BillingCustomer
    ) -> Dict[str, Any]:
        """Create customer in Chargebee"""

        # Chargebee API implementation
        # This would use Chargebee's customer creation API

        return {"id": f"chargebee_{customer.customer_id}"}

    async def create_subscription(
        self,
        customer_id: str,
        plan_id: str,
        billing_interval: BillingInterval,
        amount: Decimal,
        trial_days: int = 0,
        provider: BillingProvider = BillingProvider.STRIPE,
    ) -> Subscription:
        """Create a new subscription"""

        customer = self.customers.get(customer_id)
        if not customer:
            raise ValueError(f"Customer not found: {customer_id}")

        subscription_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Calculate billing periods
        if billing_interval == BillingInterval.MONTHLY:
            period_end = now + timedelta(days=30)
        elif billing_interval == BillingInterval.QUARTERLY:
            period_end = now + timedelta(days=90)
        elif billing_interval == BillingInterval.YEARLY:
            period_end = now + timedelta(days=365)
        else:
            period_end = now + timedelta(days=30)  # Default

        subscription = Subscription(
            subscription_id=subscription_id,
            customer_id=customer_id,
            tenant_id=customer.tenant_id,
            plan_id=plan_id,
            billing_interval=billing_interval,
            amount=amount,
            current_period_start=now,
            current_period_end=period_end,
        )

        # Set up trial if specified
        if trial_days > 0:
            subscription.trial_start = now
            subscription.trial_end = now + timedelta(days=trial_days)
            subscription.status = "trialing"

        # Create subscription in billing provider
        if provider == BillingProvider.STRIPE:
            provider_sub = await self._create_stripe_subscription(
                subscription, customer
            )
            subscription.provider_subscription_id = provider_sub.get("id")

        # Store subscription
        self.subscriptions[subscription_id] = subscription

        logger.info(
            f"Created subscription: {subscription_id} for customer: {customer_id}"
        )

        return subscription

    async def _create_stripe_subscription(
        self, subscription: Subscription, customer: BillingCustomer
    ) -> Dict[str, Any]:
        """Create subscription in Stripe"""

        # This would create a subscription in Stripe
        # For demo purposes, return mock data

        return {
            "id": f"sub_stripe_{subscription.subscription_id}",
            "status": subscription.status,
            "current_period_start": int(subscription.current_period_start.timestamp()),
            "current_period_end": int(subscription.current_period_end.timestamp()),
        }

    async def record_usage(
        self,
        subscription_id: str,
        feature: str,
        quantity: Decimal,
        timestamp: Optional[datetime] = None,
    ):
        """Record usage for a subscription"""

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        usage_record = {
            "subscription_id": subscription_id,
            "feature": feature,
            "quantity": quantity,
            "timestamp": timestamp,
            "recorded_at": datetime.now(timezone.utc),
        }

        self.usage_buffer.append(usage_record)

        # If buffer is full, process immediately
        if len(self.usage_buffer) >= 1000:
            await self._process_usage_buffer()

    async def _process_usage_buffer(self):
        """Process accumulated usage records"""

        if not self.usage_buffer:
            return

        # Group usage by subscription and feature
        usage_by_subscription = {}

        for record in self.usage_buffer:
            sub_id = record["subscription_id"]
            feature = record["feature"]
            quantity = record["quantity"]

            if sub_id not in usage_by_subscription:
                usage_by_subscription[sub_id] = {}

            if feature not in usage_by_subscription[sub_id]:
                usage_by_subscription[sub_id][feature] = Decimal("0")

            usage_by_subscription[sub_id][feature] += quantity

        # Update subscription usage
        for sub_id, features in usage_by_subscription.items():
            subscription = self.subscriptions.get(sub_id)
            if subscription:
                for feature, quantity in features.items():
                    if feature not in subscription.metered_usage:
                        subscription.metered_usage[feature] = Decimal("0")
                    subscription.metered_usage[feature] += quantity

        # Clear buffer
        self.usage_buffer.clear()

        logger.info(f"Processed usage for {len(usage_by_subscription)} subscriptions")

    async def generate_invoice(
        self, subscription_id: str, period_start: datetime, period_end: datetime
    ) -> Invoice:
        """Generate invoice for subscription period"""

        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            raise ValueError(f"Subscription not found: {subscription_id}")

        invoice_id = str(uuid.uuid4())

        # Calculate line items
        line_items = []
        total_amount = Decimal("0")

        # Base subscription amount
        if subscription.billing_interval != BillingInterval.USAGE_BASED:
            line_items.append(
                {
                    "description": f"Subscription - {subscription.plan_id}",
                    "quantity": 1,
                    "unit_price": subscription.amount,
                    "total": subscription.amount,
                }
            )
            total_amount += subscription.amount

        # Usage-based charges
        for feature, usage_quantity in subscription.metered_usage.items():
            if usage_quantity > 0:
                # Get pricing for feature
                unit_price = await self._get_feature_pricing(
                    subscription.plan_id, feature
                )
                line_total = usage_quantity * unit_price

                line_items.append(
                    {
                        "description": f"Usage - {feature}",
                        "quantity": float(usage_quantity),
                        "unit_price": float(unit_price),
                        "total": float(line_total),
                    }
                )
                total_amount += line_total

        invoice = Invoice(
            invoice_id=invoice_id,
            subscription_id=subscription_id,
            customer_id=subscription.customer_id,
            tenant_id=subscription.tenant_id,
            status=InvoiceStatus.OPEN,
            amount_due=total_amount,
            line_items=line_items,
        )

        # Store invoice
        self.invoices[invoice_id] = invoice

        # Create invoice in billing provider
        await self._create_provider_invoice(invoice)

        logger.info(f"Generated invoice: {invoice_id} for ${total_amount}")

        return invoice

    async def _get_feature_pricing(self, plan_id: str, feature: str) -> Decimal:
        """Get pricing for a feature"""

        # This would lookup pricing from configuration
        # For demo purposes, return sample pricing

        pricing_map = {
            "api_calls": Decimal("0.001"),
            "multimodal_processing": Decimal("0.01"),
            "collaboration_minutes": Decimal("0.05"),
            "storage_gb": Decimal("0.10"),
            "federated_learning_hours": Decimal("1.00"),
        }

        return pricing_map.get(feature, Decimal("0.001"))

    async def _create_provider_invoice(self, invoice: Invoice):
        """Create invoice in billing provider"""

        # Create in Stripe, Paddle, etc.
        # For demo purposes, just log

        logger.info(f"Creating provider invoice for {invoice.invoice_id}")

    async def process_payment(
        self, invoice_id: str, payment_method_id: Optional[str] = None
    ) -> PaymentStatus:
        """Process payment for an invoice"""

        invoice = self.invoices.get(invoice_id)
        if not invoice:
            raise ValueError(f"Invoice not found: {invoice_id}")

        # Process payment with billing provider
        payment_result = await self._process_provider_payment(
            invoice, payment_method_id
        )

        if payment_result["status"] == "succeeded":
            invoice.status = InvoiceStatus.PAID
            invoice.amount_paid = invoice.amount_due
            invoice.paid_at = datetime.now(timezone.utc)

            # Reset usage counters for subscription
            subscription = self.subscriptions.get(invoice.subscription_id)
            if subscription:
                subscription.metered_usage.clear()

            return PaymentStatus.SUCCEEDED
        else:
            return PaymentStatus.FAILED

    async def _process_provider_payment(
        self, invoice: Invoice, payment_method_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process payment with billing provider"""

        # Mock payment processing
        # In real implementation, this would call Stripe, Paddle, etc.

        return {
            "status": "succeeded",
            "payment_id": f"pay_{uuid.uuid4()}",
            "amount": float(invoice.amount_due),
        }

    async def handle_failed_payment(self, invoice_id: str):
        """Handle failed payment with dunning management"""

        invoice = self.invoices.get(invoice_id)
        if not invoice:
            return

        subscription = self.subscriptions.get(invoice.subscription_id)
        if not subscription:
            return

        # Update subscription status
        subscription.status = "past_due"
        subscription.updated_at = datetime.now(timezone.utc)

        # Implement dunning logic
        retry_attempts = subscription.provider_data.get("retry_attempts", 0)
        max_retries = 3

        if retry_attempts < max_retries:
            # Schedule retry
            subscription.provider_data["retry_attempts"] = retry_attempts + 1
            subscription.provider_data["next_retry"] = datetime.now(
                timezone.utc
            ) + timedelta(days=3)

            logger.warning(
                f"Payment failed for invoice {invoice_id}, retry {retry_attempts + 1}/{max_retries}"
            )
        else:
            # Cancel subscription after max retries
            subscription.status = "cancelled"
            subscription.cancelled_at = datetime.now(timezone.utc)

            logger.error(
                f"Subscription {subscription.subscription_id} cancelled due to payment failures"
            )

    async def cancel_subscription(
        self, subscription_id: str, immediate: bool = False
    ) -> bool:
        """Cancel a subscription"""

        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return False

        if immediate:
            subscription.status = "cancelled"
            subscription.cancelled_at = datetime.now(timezone.utc)
        else:
            # Cancel at period end
            subscription.status = "cancelled"
            subscription.cancelled_at = subscription.current_period_end

        subscription.updated_at = datetime.now(timezone.utc)

        # Cancel in billing provider
        await self._cancel_provider_subscription(subscription)

        logger.info(f"Cancelled subscription: {subscription_id}")

        return True

    async def _cancel_provider_subscription(self, subscription: Subscription):
        """Cancel subscription in billing provider"""

        # Cancel in Stripe, Paddle, etc.
        logger.info(
            f"Cancelling provider subscription for {subscription.subscription_id}"
        )

    async def get_billing_analytics(
        self,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get billing analytics and revenue metrics"""

        if not start_date:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if not end_date:
            end_date = datetime.now(timezone.utc)

        # Filter data
        subscriptions = list(self.subscriptions.values())
        invoices = list(self.invoices.values())

        if tenant_id:
            subscriptions = [s for s in subscriptions if s.tenant_id == tenant_id]
            invoices = [i for i in invoices if i.tenant_id == tenant_id]

        # Calculate metrics
        total_revenue = sum(
            i.amount_paid
            for i in invoices
            if i.paid_at and start_date <= i.paid_at <= end_date
        )
        outstanding_revenue = sum(
            i.amount_due - i.amount_paid
            for i in invoices
            if i.status == InvoiceStatus.OPEN
        )

        active_subscriptions = len([s for s in subscriptions if s.status == "active"])
        trial_subscriptions = len([s for s in subscriptions if s.status == "trialing"])
        cancelled_subscriptions = len(
            [s for s in subscriptions if s.status == "cancelled"]
        )

        # Monthly recurring revenue (MRR)
        monthly_revenue = sum(
            s.amount
            for s in subscriptions
            if s.status == "active" and s.billing_interval == BillingInterval.MONTHLY
        )

        # Annual recurring revenue (ARR)
        annual_revenue = monthly_revenue * 12

        analytics = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "revenue": {
                "total_revenue": float(total_revenue),
                "outstanding_revenue": float(outstanding_revenue),
                "monthly_recurring_revenue": float(monthly_revenue),
                "annual_recurring_revenue": float(annual_revenue),
            },
            "subscriptions": {
                "active": active_subscriptions,
                "trial": trial_subscriptions,
                "cancelled": cancelled_subscriptions,
                "total": len(subscriptions),
            },
            "invoices": {
                "total": len(invoices),
                "paid": len([i for i in invoices if i.status == InvoiceStatus.PAID]),
                "open": len([i for i in invoices if i.status == InvoiceStatus.OPEN]),
                "overdue": len(
                    [
                        i
                        for i in invoices
                        if i.status == InvoiceStatus.OPEN
                        and i.due_date < datetime.now(timezone.utc)
                    ]
                ),
            },
        }

        return analytics

    async def _usage_aggregation_task(self):
        """Background task for usage aggregation"""

        while True:
            try:
                await asyncio.sleep(self.usage_aggregation_interval)
                await self._process_usage_buffer()
            except Exception as e:
                logger.error(f"Error in usage aggregation task: {e}")

    async def _invoice_generation_task(self):
        """Background task for invoice generation"""

        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                await self._generate_periodic_invoices()
            except Exception as e:
                logger.error(f"Error in invoice generation task: {e}")

    async def _generate_periodic_invoices(self):
        """Generate invoices for subscriptions due for billing"""

        now = datetime.now(timezone.utc)

        for subscription in self.subscriptions.values():
            if (
                subscription.status == "active"
                and subscription.current_period_end <= now
            ):

                # Generate invoice for this billing period
                await self.generate_invoice(
                    subscription.subscription_id,
                    subscription.current_period_start,
                    subscription.current_period_end,
                )

                # Update subscription period
                period_length = (
                    subscription.current_period_end - subscription.current_period_start
                )
                subscription.current_period_start = subscription.current_period_end
                subscription.current_period_end = (
                    subscription.current_period_end + period_length
                )
                subscription.updated_at = now

    async def _payment_retry_task(self):
        """Background task for payment retries"""

        while True:
            try:
                await asyncio.sleep(3600)  # Run hourly
                await self._process_payment_retries()
            except Exception as e:
                logger.error(f"Error in payment retry task: {e}")

    async def _process_payment_retries(self):
        """Process payment retries for failed payments"""

        now = datetime.now(timezone.utc)

        for subscription in self.subscriptions.values():
            if (
                subscription.status == "past_due"
                and subscription.provider_data.get("next_retry")
                and datetime.fromisoformat(subscription.provider_data["next_retry"])
                <= now
            ):

                # Find unpaid invoices for this subscription
                unpaid_invoices = [
                    i
                    for i in self.invoices.values()
                    if (
                        i.subscription_id == subscription.subscription_id
                        and i.status == InvoiceStatus.OPEN
                    )
                ]

                for invoice in unpaid_invoices:
                    payment_status = await self.process_payment(invoice.invoice_id)

                    if payment_status == PaymentStatus.SUCCEEDED:
                        # Payment succeeded, reactivate subscription
                        subscription.status = "active"
                        subscription.provider_data.pop("retry_attempts", None)
                        subscription.provider_data.pop("next_retry", None)
                        subscription.updated_at = now
                        break
                    else:
                        # Payment failed again
                        await self.handle_failed_payment(invoice.invoice_id)


class WebhookHandler:
    """Handle webhooks from billing providers"""

    def __init__(self, billing_manager: BillingManager):
        self.billing_manager = billing_manager

    async def handle_stripe_webhook(
        self, payload: bytes, signature: str
    ) -> Dict[str, Any]:
        """Handle Stripe webhook"""

        # Verify webhook signature
        webhook_secret = self.billing_manager.stripe_config.get("webhook_secret")
        if not self._verify_stripe_signature(payload, signature, webhook_secret):
            raise ValueError("Invalid webhook signature")

        event = json.loads(payload.decode("utf-8"))
        event_type = event["type"]

        if event_type == "invoice.payment_succeeded":
            await self._handle_payment_succeeded(event["data"]["object"])
        elif event_type == "invoice.payment_failed":
            await self._handle_payment_failed(event["data"]["object"])
        elif event_type == "customer.subscription.updated":
            await self._handle_subscription_updated(event["data"]["object"])
        elif event_type == "customer.subscription.deleted":
            await self._handle_subscription_cancelled(event["data"]["object"])

        return {"status": "handled"}

    def _verify_stripe_signature(
        self, payload: bytes, signature: str, secret: str
    ) -> bool:
        """Verify Stripe webhook signature"""

        try:
            expected_signature = hmac.new(
                secret.encode("utf-8"), payload, hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature, f"sha256={expected_signature}")
        except Exception:
            return False

    async def _handle_payment_succeeded(self, stripe_invoice: Dict[str, Any]):
        """Handle successful payment"""

        # Find our invoice by provider ID
        provider_invoice_id = stripe_invoice["id"]

        for invoice in self.billing_manager.invoices.values():
            if invoice.provider_invoice_id == provider_invoice_id:
                invoice.status = InvoiceStatus.PAID
                invoice.amount_paid = Decimal(str(stripe_invoice["amount_paid"] / 100))
                invoice.paid_at = datetime.now(timezone.utc)
                break

    async def _handle_payment_failed(self, stripe_invoice: Dict[str, Any]):
        """Handle failed payment"""

        provider_invoice_id = stripe_invoice["id"]

        for invoice in self.billing_manager.invoices.values():
            if invoice.provider_invoice_id == provider_invoice_id:
                await self.billing_manager.handle_failed_payment(invoice.invoice_id)
                break

    async def _handle_subscription_updated(self, stripe_subscription: Dict[str, Any]):
        """Handle subscription update"""

        # Update subscription status based on Stripe data
        provider_subscription_id = stripe_subscription["id"]

        for subscription in self.billing_manager.subscriptions.values():
            if subscription.provider_subscription_id == provider_subscription_id:
                subscription.status = stripe_subscription["status"]
                subscription.updated_at = datetime.now(timezone.utc)
                break

    async def _handle_subscription_cancelled(self, stripe_subscription: Dict[str, Any]):
        """Handle subscription cancellation"""

        provider_subscription_id = stripe_subscription["id"]

        for subscription in self.billing_manager.subscriptions.values():
            if subscription.provider_subscription_id == provider_subscription_id:
                subscription.status = "cancelled"
                subscription.cancelled_at = datetime.now(timezone.utc)
                subscription.updated_at = datetime.now(timezone.utc)
                break
