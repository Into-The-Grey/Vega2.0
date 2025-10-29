# Module-Specific Documentation

This directory contains detailed documentation for specific Vega 2.0 modules and subsystems.

## Core Modules

### User Profiling

- **Location**: `src/vega/user/user_profiling/`
- **Documentation**: [User Profiling README](../../src/vega/user/user_profiling/README.md)
- User behavior analysis and profiling system

### Federated Learning

- **Location**: `src/vega/federated/`
- **Status Documents**:
  - [Communication Efficient Protocols Summary](../../src/vega/federated/COMMUNICATION_EFFICIENT_PROTOCOLS_SUMMARY.md)
  - [Communication Protocols Status](../../src/vega/federated/COMMUNICATION_PROTOCOLS_STATUS.md)
- Advanced federated learning implementation

## Developer Notes (DEV_NOTES.md)

Developer notes are maintained within each module's source directory for easy access during development:

- **[Core Module DEV_NOTES](../../src/vega/core/DEV_NOTES.md)** - Core application development notes
- **[Voice Module DEV_NOTES](../../src/vega/voice/DEV_NOTES.md)** - Voice processing system notes
- **[Learning Module DEV_NOTES](../../src/vega/learning/DEV_NOTES.md)** - Machine learning module notes  
- **[Training Module DEV_NOTES](../../src/vega/training/DEV_NOTES.md)** - Model training notes
- **[Integrations Module DEV_NOTES](../../src/vega/integrations/DEV_NOTES.md)** - External integrations notes
- **[Datasets Module DEV_NOTES](../../src/vega/datasets/DEV_NOTES.md)** - Dataset processing notes

## Why In-Source Documentation?

Module-specific READMEs and DEV_NOTES are kept in the source tree because they:

- Are tightly coupled to the code
- Change frequently with implementation details
- Are most useful when viewed alongside the source
- Serve as inline reference for developers

For higher-level documentation about these modules, see:

- [Architecture Overview](../architecture/ARCHITECTURE.md)
- [Development Guide](../development/)
- [Features Documentation](../features/)
