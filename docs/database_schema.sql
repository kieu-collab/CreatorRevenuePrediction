-- PostgreSQL-oriented normalized schema. Keep raw imports immutable and build
-- the model view from lagged/pre-campaign attributes only.

CREATE TABLE creators (
    creator_id TEXT PRIMARY KEY,
    creator_name TEXT NOT NULL,
    handle TEXT UNIQUE,
    primary_niche TEXT,
    tiktok_url TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE creator_snapshots (
    creator_id TEXT REFERENCES creators(creator_id),
    snapshot_date DATE NOT NULL,
    followers BIGINT,
    avg_views_30d NUMERIC,
    engagement_rate_30d NUMERIC,
    conversion_rate_30d NUMERIC,
    video_count_30d INTEGER,
    live_count_30d INTEGER,
    source_name TEXT,
    PRIMARY KEY (creator_id, snapshot_date)
);

CREATE TABLE brands (
    brand_id TEXT PRIMARY KEY,
    brand_name TEXT NOT NULL,
    vertical TEXT
);

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    brand_id TEXT REFERENCES brands(brand_id),
    product_name TEXT NOT NULL,
    category_l1 TEXT,
    category_l2 TEXT,
    list_price_vnd NUMERIC,
    gross_margin_rate NUMERIC
);

CREATE TABLE campaigns (
    campaign_id TEXT PRIMARY KEY,
    brand_id TEXT REFERENCES brands(brand_id),
    product_id TEXT REFERENCES products(product_id),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    discount_rate NUMERIC,
    campaign_type TEXT,
    status TEXT
);

CREATE TABLE creator_campaigns (
    campaign_id TEXT REFERENCES campaigns(campaign_id),
    creator_id TEXT REFERENCES creators(creator_id),
    snapshot_date DATE NOT NULL,
    planned_posts INTEGER,
    planned_live_sessions INTEGER,
    creator_cost_vnd NUMERIC,
    brand_fit_score NUMERIC,
    actual_views BIGINT,
    product_clicks BIGINT,
    orders BIGINT,
    attributed_gmv_vnd NUMERIC,
    attributed_revenue_vnd NUMERIC,
    PRIMARY KEY (campaign_id, creator_id)
);

-- Enforce snapshot_date < campaign.start_date in the ingestion job or a trigger;
-- PostgreSQL CHECK constraints cannot contain cross-table subqueries.
-- Training view rule: creator snapshot must precede campaign start.
-- Outcome columns are labels/monitoring fields, never features.
