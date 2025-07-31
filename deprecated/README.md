# Deprecated Components

This directory contains code and configurations that are no longer actively used in production but are retained for historical reference and potential future needs.

## Deprecation Policy

Components are moved here when they:
- Are replaced by newer implementations
- No longer fit the current architecture
- May have future reference value
- Should not be deleted immediately for safety reasons

## Current Contents

### [Date: 2025-07-31] Database Infrastructure
- **database/**: Empty database folder from initial architecture
- **Reason**: System uses in-memory storage and file system instead of persistent database
- **Status**: Safe to remove after 90 days if no database needs arise

### [Date: 2025-07-31] Development Utilities  
- **cli_video_generator.py**: Duplicate CLI interface
- **Reason**: Functionality merged into main video_generator.py
- **Status**: Redundant wrapper, can be safely removed

## Retention Guidelines

- Items remain here for minimum 90 days
- Critical components stay for 6 months
- Review quarterly and purge outdated items
- Document removal decisions in project changelog

## Recovery

All deprecated items are also available in Git history. Use this folder only for quick reference during active development phases.

---
*Last updated: July 31, 2025*
