from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool, text
from alembic import context
import os
import sys

# ✅ 1. Add project root to path so models can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ 2. Import your SQLAlchemy Base
from models import Base  # make sure this is the correct relative import

# 🔧 3. Alembic Config object (from alembic.ini)
config = context.config

# ✅ 4. Optional: Configure logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ✅ 5. Provide metadata for autogenerate support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # ✅ Pick up column type changes
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # ✅ 6. Ensure PostgreSQL extensions are available before migrations
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,       # ✅ detect column type changes (e.g., JSON, pgvector)
            render_as_batch=False    # ✅ True only for SQLite; keep False for PostgreSQL
        )

        with context.begin_transaction():
            context.run_migrations()


# ✅ 7. Main trigger
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
