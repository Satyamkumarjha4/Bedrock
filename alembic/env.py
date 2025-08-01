from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool, text
from alembic import context
import os
import sys

# ✅ 1. Add project root to path so models and utils can be discovered
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ 2. Import SQLAlchemy Base (core models)
from models import Base

# ✅ 3. (Optional future-proofing) Import VectorDBUtils if needed
# from vcare_ai.utils.vectorDB_utils import VectorDBUtils  # Uncomment only if needed later

# 🔧 4. Alembic Config object (from alembic.ini)
config = context.config

# 📜 5. Configure logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 🔁 6. Set metadata for autogenerate (used during `alembic revision --autogenerate`)
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
        # ✅ 7. Ensure PostgreSQL extensions required by pgvector & pg_trgm are present
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=False,  # ✅ Only needed for SQLite
        )

        with context.begin_transaction():
            context.run_migrations()


# ✅ 8. Execute migration in correct mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
