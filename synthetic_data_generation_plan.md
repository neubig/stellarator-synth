# Synthetic Data Generation Plan for Stellarator Simulator ML Surrogate

## Executive Summary

Based on analysis of the ConStellaration repository and VMEC++ simulator, this plan outlines a comprehensive approach to generate synthetic training data that would enable a machine learning model to mimic the physics simulators used in stellarator design.

## Current Simulator Architecture

### Open Source Status ✅
- **VMEC++**: Open source (MIT license) - C++ reimplementation of VMEC
- **ConStellaration**: Open source optimization framework and dataset tools
- **Existing Dataset**: 150,000+ QI equilibria already available on Hugging Face

### Key Simulators
1. **VMEC++**: Primary physics simulator (ideal-MHD equilibrium solver)
2. **Boozer**: Coordinate transformation tool
3. **Forward Model**: Complete pipeline integrating all components

### Input-Output Mapping
- **Input**: 3D plasma boundary surface (Fourier coefficients) + MHD parameters
- **Processing**: VMEC++ → Boozer coordinates → QI analysis → Turbulent transport
- **Output**: ~12 key metrics (aspect ratio, elongation, rotational transform, QI residuals, etc.)

## Synthetic Data Generation Strategy

### Phase 1: Systematic Parameter Space Exploration

#### 1.1 Boundary Parameter Sampling
```python
# Key parameters to vary systematically:
- aspect_ratio: [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
- elongation: [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
- rotational_transform: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
- n_field_periods: [2, 3, 4, 5, 6]
- triangularity: [-0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
```

#### 1.2 Fourier Mode Variations
- Systematically vary Fourier coefficients for boundary representation
- Include both symmetric and asymmetric configurations
- Cover different toroidal and poloidal mode numbers

#### 1.3 MHD Parameter Variations
- Pressure profiles (different shapes and magnitudes)
- Current profiles
- Vacuum field configurations

### Phase 2: Advanced Sampling Techniques

#### 2.1 Latin Hypercube Sampling
- Generate space-filling designs across the full parameter space
- Ensure good coverage of parameter combinations
- Target: 50,000-100,000 additional configurations

#### 2.2 Adaptive Sampling
- Identify regions where ML model has high uncertainty
- Generate additional data in those regions
- Use active learning principles

#### 2.3 Physics-Informed Sampling
- Focus on physically realistic configurations
- Use existing QI stellarator designs as starting points
- Apply small perturbations to known good designs

### Phase 3: Data Augmentation and Enrichment

#### 3.1 Multi-Fidelity Data Generation
```python
# Generate data at different fidelity levels:
settings_low = ConstellarationSettings(
    vmec_preset_settings=VmecPresetSettings(fidelity="low_fidelity")
)
settings_high = ConstellarationSettings(
    vmec_preset_settings=VmecPresetSettings(fidelity="high_fidelity")
)
```

#### 3.2 Failure Case Documentation
- Systematically explore parameter regions where VMEC++ fails to converge
- Document failure modes and boundary conditions
- Create classification data for convergence prediction

#### 3.3 Sensitivity Analysis Data
- Generate data with small parameter perturbations
- Compute gradients and sensitivity information
- Enable gradient-based ML training

### Phase 4: Specialized Dataset Creation

#### 4.1 Time Series Data
- Generate optimization trajectories
- Document how metrics evolve during optimization
- Enable sequence-to-sequence ML models

#### 4.2 Multi-Objective Data
- Generate Pareto fronts for different objective combinations
- Include trade-off information between competing objectives
- Support multi-objective optimization ML models

#### 4.3 Constraint Violation Data
- Systematically explore infeasible regions
- Document constraint violation patterns
- Enable constraint-aware ML models

## Implementation Plan

### Step 1: Infrastructure Setup (Week 1-2)
```python
# Create automated data generation pipeline
class SyntheticDataGenerator:
    def __init__(self, output_dir: str, n_workers: int = 8):
        self.output_dir = output_dir
        self.n_workers = n_workers
        
    def generate_systematic_grid(self, param_ranges: dict) -> None:
        # Systematic parameter space exploration
        
    def generate_lhs_samples(self, n_samples: int) -> None:
        # Latin hypercube sampling
        
    def generate_adaptive_samples(self, model, n_samples: int) -> None:
        # Adaptive sampling based on model uncertainty
```

### Step 2: Systematic Generation (Week 3-6)
- Implement grid-based parameter exploration
- Generate ~25,000 configurations covering key parameter ranges
- Focus on convergent cases first

### Step 3: Advanced Sampling (Week 7-10)
- Implement LHS and adaptive sampling
- Generate additional 50,000 configurations
- Include failure cases and edge conditions

### Step 4: Quality Assurance (Week 11-12)
- Validate data quality and coverage
- Check for duplicates and outliers
- Ensure balanced representation across parameter space

## Expected Outcomes

### Dataset Size
- **Target**: 200,000-300,000 total configurations
- **Current**: 150,000 existing + 100,000-150,000 new
- **Coverage**: Comprehensive parameter space coverage

### Data Quality
- **Convergence Rate**: >95% successful VMEC++ runs
- **Parameter Coverage**: Uniform coverage across all key parameters
- **Physics Validity**: All configurations physically meaningful

### ML Model Enablement
- **Surrogate Accuracy**: Target <1% error on key metrics
- **Speed Improvement**: 1000x+ faster than VMEC++ simulation
- **Uncertainty Quantification**: Enable confidence intervals

## Resource Requirements

### Computational Resources
- **CPU Hours**: ~10,000-20,000 CPU hours for full generation
- **Storage**: ~100-200 GB for complete dataset
- **Memory**: 16-32 GB RAM per worker process

### Timeline
- **Total Duration**: 12 weeks
- **Parallel Execution**: 8-16 worker processes
- **Incremental Delivery**: Weekly data releases

## Risk Mitigation

### Technical Risks
1. **VMEC++ Convergence Issues**: Implement robust error handling and retry logic
2. **Parameter Space Gaps**: Use adaptive sampling to fill gaps
3. **Data Quality Issues**: Implement comprehensive validation checks

### Resource Risks
1. **Computational Limits**: Prioritize most important parameter regions
2. **Storage Constraints**: Implement data compression and archiving
3. **Time Constraints**: Focus on core functionality first

## Success Metrics

### Quantitative Metrics
- **Data Volume**: 200,000+ successful simulations
- **Parameter Coverage**: >90% of feasible parameter space
- **ML Model Performance**: <1% RMSE on key metrics

### Qualitative Metrics
- **Physics Consistency**: All generated data passes physics sanity checks
- **Optimization Utility**: Data enables effective stellarator optimization
- **Community Adoption**: Dataset used by fusion research community

## Conclusion

This comprehensive synthetic data generation plan leverages the open-source VMEC++ simulator and ConStellaration framework to create a large-scale dataset suitable for training ML surrogate models. The systematic approach ensures good parameter space coverage while the adaptive sampling techniques focus computational resources on the most valuable data points.

The resulting dataset would enable the development of fast, accurate ML models that could replace expensive physics simulations in stellarator optimization workflows, potentially accelerating fusion energy research by orders of magnitude.