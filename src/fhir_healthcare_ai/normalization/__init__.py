"""Normalization layer: raw FHIR resources in, patient-centric records out."""

from fhir_healthcare_ai.normalization.assembler import RecordAssembler, assemble_records

__all__ = ["RecordAssembler", "assemble_records"]
