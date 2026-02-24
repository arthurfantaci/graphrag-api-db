#!/usr/bin/env bash
# neo4j-staging.sh — Manage local Neo4j Desktop staging instance
#
# Subcommands:
#   env      Print export statements for staging Neo4j env vars
#   status   Test connectivity to local staging DBMS
#   promote  Dump staging DB and upload to production Aura
#
# Usage:
#   eval $(./scripts/neo4j-staging.sh env)    # Switch terminal to staging
#   ./scripts/neo4j-staging.sh status         # Check staging connectivity
#   ./scripts/neo4j-staging.sh promote        # Promote staging → production

set -euo pipefail

# Staging connection defaults
STAGING_URI="bolt://localhost:7687"
STAGING_USERNAME="neo4j"
STAGING_PASSWORD="stagingpassword"
STAGING_DATABASE="neo4j"

BACKUP_DIR="./backups"

usage() {
    cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  env       Print export statements for staging Neo4j env vars
  status    Test connectivity to local staging DBMS and show node count
  promote   Dump staging database and upload to production Aura

Examples:
  eval \$(./scripts/neo4j-staging.sh env)   # Switch shell to staging
  ./scripts/neo4j-staging.sh status         # Check staging is reachable
  ./scripts/neo4j-staging.sh promote        # Promote staging to production
EOF
    exit 1
}

# ─── env ──────────────────────────────────────────────────────────────────────

cmd_env() {
    cat <<EOF
export NEO4J_URI="${STAGING_URI}"
export NEO4J_USERNAME="${STAGING_USERNAME}"
export NEO4J_PASSWORD="${STAGING_PASSWORD}"
export NEO4J_DATABASE="${STAGING_DATABASE}"
EOF
}

# ─── status ───────────────────────────────────────────────────────────────────

cmd_status() {
    echo "Checking staging DBMS at ${STAGING_URI}..."
    echo

    # Try cypher-shell first (comes with Neo4j Desktop)
    if command -v cypher-shell &>/dev/null; then
        if cypher-shell -u "${STAGING_USERNAME}" -p "${STAGING_PASSWORD}" \
            -a "${STAGING_URI}" -d "${STAGING_DATABASE}" \
            "RETURN 'connected' AS status" &>/dev/null; then
            echo "  Connected to staging DBMS"
            echo
            node_count=$(cypher-shell -u "${STAGING_USERNAME}" -p "${STAGING_PASSWORD}" \
                -a "${STAGING_URI}" -d "${STAGING_DATABASE}" \
                --format plain \
                "MATCH (n) RETURN count(n) AS nodes" 2>/dev/null | tail -1)
            echo "  Node count: ${node_count}"
        else
            echo "  ERROR: Cannot connect to staging DBMS at ${STAGING_URI}"
            echo "  Make sure the DBMS is started in Neo4j Desktop."
            exit 1
        fi
    else
        echo "  cypher-shell not found in PATH."
        echo
        echo "  Options:"
        echo "    1. Open Neo4j Desktop → DBMS '...' menu → Terminal"
        echo "       Then run: cypher-shell -u neo4j -p stagingpassword"
        echo "    2. Open Neo4j Browser (via Desktop) and run:"
        echo "       MATCH (n) RETURN count(n)"
        exit 1
    fi
}

# ─── promote ──────────────────────────────────────────────────────────────────

cmd_promote() {
    # Read production URI from .env file
    if [[ ! -f .env ]]; then
        echo "ERROR: .env file not found. Cannot determine production URI."
        exit 1
    fi

    prod_uri=$(grep -E '^NEO4J_URI=' .env | sed 's/^NEO4J_URI=//' | tr -d '"' | tr -d "'")
    if [[ -z "${prod_uri}" ]]; then
        echo "ERROR: NEO4J_URI not found in .env file."
        exit 1
    fi

    echo "=== Staging → Production Promotion ==="
    echo
    echo "  Staging:    ${STAGING_URI}"
    echo "  Production: ${prod_uri}"
    echo
    echo "  WARNING: This will OVERWRITE the production database."
    echo
    read -rp "  Continue? [y/N]: " confirm
    if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
        echo "  Aborted."
        exit 0
    fi

    echo

    # Create backup directory
    mkdir -p "${BACKUP_DIR}"
    timestamp=$(date +%Y%m%dT%H%M%S)
    dump_file="${BACKUP_DIR}/staging-${timestamp}.dump"

    # Check if neo4j-admin is available
    if ! command -v neo4j-admin &>/dev/null; then
        echo "  neo4j-admin not found in PATH."
        echo
        echo "  Manual promotion steps:"
        echo
        echo "  1. STOP the staging DBMS in Neo4j Desktop"
        echo
        echo "  2. Open Desktop Terminal (DBMS '...' menu → Terminal) and run:"
        echo "     neo4j-admin database dump neo4j --to-path=${BACKUP_DIR}"
        echo
        echo "  3. Upload to Aura Console:"
        echo "     - Go to https://console.neo4j.io"
        echo "     - Select your instance → 'Restore from backup file'"
        echo "     - Upload the .dump file from ${BACKUP_DIR}/"
        echo
        echo "  OR use neo4j-admin database upload (from Desktop Terminal):"
        echo "     neo4j-admin database upload neo4j \\"
        echo "       --to-uri=${prod_uri} \\"
        echo "       --overwrite-destination=true"
        exit 0
    fi

    # Verify staging DBMS is stopped (required for dump)
    if cypher-shell -u "${STAGING_USERNAME}" -p "${STAGING_PASSWORD}" \
        -a "${STAGING_URI}" "RETURN 1" &>/dev/null 2>&1; then
        echo "  ERROR: Staging DBMS is still running."
        echo "  Please STOP it in Neo4j Desktop before dumping."
        echo "  (Desktop → graphrag-api-db-stage → click 'Stop')"
        exit 1
    fi

    echo "  Dumping staging database to ${dump_file}..."
    neo4j-admin database dump "${STAGING_DATABASE}" --to-path="${BACKUP_DIR}"

    # Rename the dump to include timestamp
    if [[ -f "${BACKUP_DIR}/${STAGING_DATABASE}.dump" ]]; then
        mv "${BACKUP_DIR}/${STAGING_DATABASE}.dump" "${dump_file}"
    fi

    echo "  Dump complete: ${dump_file}"
    echo

    echo "  Uploading to production (${prod_uri})..."
    neo4j-admin database upload "${STAGING_DATABASE}" \
        --to-uri="${prod_uri}" \
        --overwrite-destination=true

    echo
    echo "  Promotion complete!"
    echo
    echo "  Next steps:"
    echo "    1. Restart staging DBMS in Neo4j Desktop"
    echo "    2. Open a new terminal (fresh env → production)"
    echo "    3. Run: graphrag-kg validate"
}

# ─── main ─────────────────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    usage
fi

case "$1" in
    env)     cmd_env ;;
    status)  cmd_status ;;
    promote) cmd_promote ;;
    *)       usage ;;
esac
