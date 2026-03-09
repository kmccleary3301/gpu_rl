# Register Pressure Playbook

High registers per thread and spill indicators often point to aggressive fusion, wide tiles, or excess temporary state. Compare PTX/SASS around hot spans and look for opportunities to shrink live ranges, split work, or simplify address calculations.
