from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gpu_cockpit.versions import SCHEMA_VERSION


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        use_enum_values=True,
    )
    schema_version: str = Field(default=SCHEMA_VERSION)

    @classmethod
    def schema_name(cls) -> str:
        return cls.__name__

    @classmethod
    def contract_schema_version(cls) -> str:
        return SCHEMA_VERSION
