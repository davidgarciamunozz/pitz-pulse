CREATE TABLE requests (
    id TEXT PRIMARY KEY NOT NULL,
    raw_message TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('processing', 'completed', 'failed')),
    ai_classification_json TEXT,
    human_correction_json TEXT,
    provider_metadata_json TEXT,
    effective_category TEXT CHECK (effective_category IS NULL OR effective_category IN
        ('bug', 'datos', 'acceso', 'automatizacion', 'consulta', 'otro')),
    effective_priority TEXT CHECK (effective_priority IS NULL OR effective_priority IN
        ('alta', 'media', 'baja')),
    effective_area TEXT CHECK (effective_area IS NULL OR effective_area IN
        ('backend', 'frontend', 'data', 'devops', 'producto', 'digital_transformation')),
    failure_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    classified_at TEXT,
    corrected_at TEXT,
    CHECK (state != 'completed' OR
        (ai_classification_json IS NOT NULL AND provider_metadata_json IS NOT NULL
         AND classified_at IS NOT NULL AND failure_code IS NULL
         AND effective_category IS NOT NULL AND effective_priority IS NOT NULL
         AND effective_area IS NOT NULL)),
    CHECK (state = 'completed' OR
        (ai_classification_json IS NULL AND human_correction_json IS NULL
         AND provider_metadata_json IS NULL AND classified_at IS NULL)),
    CHECK (human_correction_json IS NULL OR corrected_at IS NOT NULL),
    CHECK (state != 'failed' OR failure_code IS NOT NULL),
    CHECK (state != 'processing' OR failure_code IS NULL)
);

CREATE INDEX requests_completed_order
    ON requests (state, created_at DESC, id DESC);
