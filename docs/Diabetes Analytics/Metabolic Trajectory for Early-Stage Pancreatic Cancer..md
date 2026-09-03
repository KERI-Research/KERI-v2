# Metabolic Trajectory for Early-Stage Pancreatic Cancer

Pancreatic cancer is notoriously insidious, rapidly progressive, and typically diagnosed at an advanced, unresectable stage, which contributes to a low global 5-year survival rate of approximately 13%. However, because pancreatic cancer is closely linked to disordered glucose metabolism, **New-Onset Diabetes Mellitus (NODM)** serves as a critical early clinical signal and screening filter.

Dysglycemia or diabetes typically develops within 3 years prior to a pancreatic cancer diagnosis, providing a high-yield diagnostic window to catch the disease in its early, resectable stages. This condition, often termed **pancreatogenic diabetes** (type 3c diabetes or PC-related diabetes mellitus, PCDM), is caused directly by the tumor's diabetogenic effects rather than standard metabolic pathways.

Tracking **metabolic trajectories** over time helps detect early-stage pancreatic cancer through several key mechanisms:

## 1. Capturing Longitudinal Temporal Dynamics

Rather than relying on static, single risk factors, analyzing how metabolic variables evolve over time helps capture the systemic changes induced by early tumor development. For example, the Mayo Clinic’s **END-PAC** score evaluates trajectories based on three longitudinal metrics: **age, weight change, and glycemic change** (tracking fluctuations over a 12-month period). Evolving metabolic trends are far more indicative of an underlying occult malignancy than isolated biomarkers, allowing clinicians to enrich the detection rate of pancreatic cancer in the high-risk diabetic population.

## 2. Identifying Coordinated Perturbations on the Inflammation-Lipid-Insulin Axis

Advanced proteomics and metabolomics reveal that early-stage pancreatic cancer-induced diabetes causes coordinated perturbations along a systemic **inflammation-lipid metabolism-insulin signaling axis**. This dysregulation distinguishes PC-related diabetes from typical Type 2 diabetes (T2DM). Researchers have validated three key blood-based protein biomarkers that reflect this pathogenic axis:

* **PLTP** (Phospholipid Transfer Protein): Facilitates the lipid remodeling critical for tumor adaptation.
* **CRTAC1** (Cartilage Acidic Protein 1): Strongly linked to body mass index (BMI) and reflects a stromally mediated metabolic support system.
* **ITGAV** (Integrin Subunit Alpha V): Drives tumor metastasis and fibrosis-associated insulin resistance.

Crucially, the expression of these proteins remains stable between early-stage and late-stage pancreatic tumors, making them highly reliable **early-stage diagnostic biomarkers**. Additionally, other tumor-secreted diabetogenic factors like **galectin-3** and **S100A9** can further assist in differentiating PC-related diabetes from typical Type 2 diabetes.

## 3. Multi-Dimensional Machine Learning Risk Stratification

By feeding multi-dimensional clinical, biochemical, and anthropometric variables into **machine learning models** (such as CatBoost), clinicians can identify high-risk individuals with high accuracy. A validated 14-variable model utilizing baseline parameters—including **glycated hemoglobin (HbA1c)**, **total cholesterol**, **age**, **trunk fat mass**, and functional phenotypes like **usual walking pace**—effectively separates high-risk and low-risk cases. At an optimal risk threshold (e.g. 0.257), such models achieve a high sensitivity of **87.7%**, minimizing missed diagnoses while successfully ruling out nearly 70% of low-risk individuals to reduce unnecessary imaging burdens.

## 4. Differentiating Genetic Susceptibility

Patients with pancreatic cancer-induced diabetes fundamentally differ from those with standard T2DM in their genetic susceptibility. UK Biobank data reveals that NODM patients who develop pancreatic cancer have a higher genetic risk for pancreatic cancer (**PC PRS**) but an inverse, lower genetic risk for Type 2 diabetes itself (**T2DM PRS**).

Combining these distinct genetic profiles into a unified **NODM-PC Polygenic Risk Score (PRS)** alongside clinical demographics yields a robust risk prediction tool (achieving a Harrell’s C-index of 0.823). This genetic-clinical approach significantly improves early reclassification, ensuring that patients on a trajectory toward cancer are prioritised for second-line diagnostic screenings, such as CT or MRI scans.
