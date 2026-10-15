
% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1:  17]) = 'Serpent pre-2.2.1' ;
COMPILE_DATE              (idx, [1:  20]) = 'Jul 11 2024 22:55:21' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:   8]) = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1:  28]) = 'gFHR_equilibrium_100.serpent' ;
WORKING_DIRECTORY         (idx, [1:  66]) = '/global/scratch/users/ikolaja/PEARLSim_dev/PEARLSim/gFHR_benchmark' ;
HOSTNAME                  (idx, [1:  12]) = 'n0024.savio4' ;
CPU_TYPE                  (idx, [1:  40]) = 'Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz' ;
CPU_MHZ                   (idx, 1)        = 218104675.0 ;
START_DATE                (idx, [1:  24]) = 'Sat Nov 15 03:56:21 2025' ;
COMPLETE_DATE             (idx, [1:  24]) = 'Sat Nov 15 03:59:58 2025' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 35 ;
SKIP                      (idx, 1)        = 15 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 42 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000 ;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ] ;
B1_IMPLICIT_LEAKAGE       (idx, 1)        = 0 ;
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 0 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 2 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 0 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 1 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 56 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_HISTORY_PROFILE       (idx, [1:  56]) = [  1.04332E+00  1.03987E+00  1.02749E+00  1.03251E+00  1.01033E+00  1.03050E+00  9.83348E-01  1.02694E+00  9.70639E-01  9.68633E-01  1.02616E+00  9.79335E-01  9.79558E-01  9.87027E-01  9.51242E-01  9.70639E-01  9.77997E-01  1.02950E+00  1.02370E+00  1.03976E+00  1.02905E+00  1.02950E+00  1.03162E+00  1.02783E+00  1.02616E+00  1.01490E+00  1.03296E+00  1.02381E+00  9.76659E-01  9.64285E-01  9.61387E-01  9.75545E-01  1.02359E+00  9.77774E-01  9.85020E-01  9.83459E-01  9.47786E-01  1.02337E+00  1.03652E+00  9.81564E-01  1.02237E+00  1.01991E+00  1.03652E+00  9.66626E-01  9.79781E-01  9.58377E-01  1.02147E+00  9.99735E-01  9.74764E-01  9.78554E-01  9.69413E-01  9.77217E-01  1.03463E+00  9.68521E-01  9.59380E-01  9.81453E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 0 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1:  71]) = '/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1:  67]) = '/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1:   3]) = 'N/A' ;
NFY_DATA_FILE_PATH        (idx, [1:  67]) = '/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:   3]) = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:   2]) = [  9.00000E-01  9.00000E-01 ] ;
ST_FRAC                   (idx, [1:   4]) = [  8.74226E-01 0.00014  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  1.25774E-01 0.00096  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  2.64718E-01 0.00025  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  9.92677E-01 2.0E-05  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  5.11105E-01 0.00080  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  3.28959E+01 0.00158  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
CELL_SEARCH_FRAC          (idx, [1:  10]) = [  9.82184E-01 7.0E-06  1.76948E-02 0.00038  1.21212E-04 0.00149  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  1.62737E+02 0.00166  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  1.62720E+02 0.00166  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  1.53311E+02 0.00041  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.34181E+03 0.00118  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 35 ;
SIMULATED_HISTORIES       (idx, 1)        = 349956 ;
MEAN_POP_SIZE             (idx, [1:   2]) = [  9.99874E+03 0.00263 ] ;
MEAN_POP_WGT              (idx, [1:   2]) = [  9.99874E+03 0.00263 ] ;
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.93774E+01 ;
RUNNING_TIME              (idx, 1)        =  3.61915E+00 ;
INIT_TIME                 (idx, [1:   2]) = [  2.10402E+00  2.10402E+00 ] ;
PROCESS_TIME              (idx, [1:   2]) = [  3.38000E-02  3.38000E-02 ] ;
TRANSPORT_CYCLE_TIME      (idx, [1:   3]) = [  1.47997E+00  1.47997E+00  0.00000E+00 ] ;
BURNUP_CYCLE_TIME         (idx, [1:   2]) = [  0.00000E+00  0.00000E+00 ] ;
BATEMAN_SOLUTION_TIME     (idx, [1:   2]) = [  0.00000E+00  0.00000E+00 ] ;
MPI_OVERHEAD_TIME         (idx, [1:   2]) = [  0.00000E+00  0.00000E+00 ] ;
ESTIMATED_RUNNING_TIME    (idx, [1:   2]) = [  3.61778E+00  0.00000E+00 ] ;
CPU_USAGE                 (idx, 1)        = 8.11720 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  5.37996E+01 0.02800 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  1.26452E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 515428.37 ;
ALLOC_MEMSIZE             (idx, 1)        = 47337.82 ;
MEMSIZE                   (idx, 1)        = 46829.59 ;
XS_MEMSIZE                (idx, 1)        = 20248.30 ;
MAT_MEMSIZE               (idx, 1)        = 893.17 ;
RES_MEMSIZE               (idx, 1)        = 2.61 ;
IFC_MEMSIZE               (idx, 1)        = 0.00 ;
MISC_MEMSIZE              (idx, 1)        = 25685.51 ;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00 ;
UNUSED_MEMSIZE            (idx, 1)        = 508.24 ;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 1309 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 272467 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-11 ;
NEUTRON_EMAX              (idx, 1)        =  2.00000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.10000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 232 ;
URES_USED                 (idx, 1)        = 113 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 2655 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 1520 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1135 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 34258 ;
TOT_TRANSMU_REA           (idx, 1)        = 1727 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:   9]) = [ 0 0 0 0 0 0 0 0 0 ] ;
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  5.22086E+19 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.71961E+07 ;
TOT_SF_RATE               (idx, 1)        =  7.84440E+05 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  6.07701E+18 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  4.25369E+05 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61316E+19 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.67707E+07 ;
INHALATION_TOXICITY       (idx, 1)        =  2.89711E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  2.26680E+10 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.81845E+09 ;
ACTINIDE_ING_TOX          (idx, 1)        =  2.52410E+09 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.31526E+10 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.01439E+10 ;
SR90_ACTIVITY             (idx, 1)        =  1.92309E+15 ;
TE132_ACTIVITY            (idx, 1)        =  3.73436E+17 ;
I131_ACTIVITY             (idx, 1)        =  2.47976E+17 ;
I132_ACTIVITY             (idx, 1)        =  3.77698E+17 ;
CS134_ACTIVITY            (idx, 1)        =  4.27044E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.04406E+15 ;
PHOTON_DECAY_SOURCE       (idx, [1:   2]) = [  4.44529E+19  8.09509E+06 ] ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  9.13826E+16 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.60692E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  6.26700E+19 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.73427E+15 0.00168  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 320 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:   3]) = [ 0 0 0 ] ;
BURNUP                    (idx, [1:   2]) = [  0.00000E+00  0.00000E+00 ] ;
BURN_DAYS                 (idx, [1:   2]) = [  0.00000E+00  0.00000E+00 ] ;
FIMA                      (idx, [1:   3]) = [  0.00000E+00  0.00000E+00  2.09931E+27 ] ;

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  2.91947E-01 0.00428 ];
U235_FISS                 (idx, [1:   4]) = [  7.77100E+18 0.00071  9.01883E-01 0.00070 ];
U238_FISS                 (idx, [1:   4]) = [  1.76744E+16 0.05603  2.05125E-03 0.05603 ];
PU239_FISS                (idx, [1:   4]) = [  8.09360E+17 0.00686  9.39326E-02 0.00687 ];
PU240_FISS                (idx, [1:   4]) = [  4.93936E+13 1.00000  5.73263E-06 1.00000 ];
PU241_FISS                (idx, [1:   4]) = [  1.77417E+16 0.05166  2.05906E-03 0.05166 ];
U235_CAPT                 (idx, [1:   4]) = [  1.93061E+18 0.00565  2.25175E-01 0.00418 ];
U238_CAPT                 (idx, [1:   4]) = [  3.06572E+18 0.00454  3.57605E-01 0.00353 ];
PU239_CAPT                (idx, [1:   4]) = [  4.88435E+17 0.01007  5.69951E-02 0.01068 ];
PU240_CAPT                (idx, [1:   4]) = [  1.52670E+17 0.01852  1.78061E-02 0.01819 ];
PU241_CAPT                (idx, [1:   4]) = [  6.73717E+15 0.10469  7.84949E-04 0.10465 ];
XE135_CAPT                (idx, [1:   4]) = [  4.55246E+17 0.00981  5.31046E-02 0.00947 ];
XE135M_CAPT               (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC      (idx, [1:   2]) = [ 0 0.00000E+00 ] ;
BALA_SRC_NEUTRON_FISS     (idx, [1:   2]) = [ 349956 3.50000E+05 ] ;
BALA_SRC_NEUTRON_NXN      (idx, [1:   2]) = [ 0 2.83330E+03 ] ;
BALA_SRC_NEUTRON_VR       (idx, [1:   2]) = [ 0 0.00000E+00 ] ;
BALA_SRC_NEUTRON_TOT      (idx, [1:   2]) = [ 349956 3.52833E+05 ] ;

BALA_LOSS_NEUTRON_CAPT    (idx, [1:   2]) = [ 171655 1.73007E+05 ] ;
BALA_LOSS_NEUTRON_FISS    (idx, [1:   2]) = [ 172433 1.73908E+05 ] ;
BALA_LOSS_NEUTRON_LEAK    (idx, [1:   2]) = [ 5868 5.91877E+03 ] ;
BALA_LOSS_NEUTRON_CUT     (idx, [1:   2]) = [ 0 0.00000E+00 ] ;
BALA_LOSS_NEUTRON_ERR     (idx, [1:   2]) = [ 0 0.00000E+00 ] ;
BALA_LOSS_NEUTRON_TOT     (idx, [1:   2]) = [ 349956 3.52833E+05 ] ;

BALA_NEUTRON_DIFF         (idx, [1:   2]) = [ 0 -6.17001E-09 ] ;

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  2.80000E+08 0.0E+00  2.80000E+08 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.38165E-01 0.0E+00  3.38165E-01 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.13576E+19 9.2E-05  2.13576E+19 9.2E-05  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  8.61642E+18 1.7E-05  8.61642E+18 1.7E-05  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  8.57338E+18 0.00336  6.46948E+18 0.00331  2.10391E+18 0.00554 ];
TOT_ABSRATE               (idx, [1:   6]) = [  1.71898E+19 0.00168  1.50859E+19 0.00142  2.10391E+18 0.00554 ];
TOT_SRCRATE               (idx, [1:   6]) = [  1.73427E+19 0.00168  1.73427E+19 0.00168  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  7.52728E+21 0.00233  3.22177E+19 0.00185  7.49506E+21 0.00234 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.93302E+17 0.01088 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  1.74831E+19 0.00169 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  2.82242E+21 0.00248 ];
INI_FMASS                 (idx, 1)        =  8.27997E+02 ;
TOT_FMASS                 (idx, 1)        =  8.27997E+02 ;
INI_BURN_FMASS            (idx, 1)        =  8.27997E+02 ;
TOT_BURN_FMASS            (idx, 1)        =  8.27997E+02 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.84719E+00 0.00115 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.40163E-01 0.00095 ];
SIX_FF_P                  (idx, [1:   2]) = [  6.61565E-01 0.00123 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.22026E+00 0.00101 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99120E-01 4.5E-05 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.83956E-01 0.00018 ];
SIX_FF_KINF               (idx, [1:   2]) = [  1.25281E+00 0.00167 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  1.23162E+00 0.00168 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.47871E+00 0.00011 ];
FISSE                     (idx, [1:   2]) = [  2.02825E+02 1.7E-05 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.23305E+00 0.00170  1.22400E+00 0.00167  7.62212E-03 0.02729 ];
COL_KEFF                  (idx, [1:   2]) = [  1.23162E+00 0.00168 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 2.000000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  1.79462E+01 0.00031 ];
IMP_ALF                   (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  3.21592E-07 0.00561 ];
IMP_EALF                  (idx, [1:   2]) = [  2.00000E+01 0.0E+00 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  9.97608E-03 0.04329 ];
IMP_AFGE                  (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.81154E-03 0.01760  1.81029E-04 0.09183  7.76135E-04 0.05166  7.68724E-04 0.05209  2.19992E-03 0.02857  6.62981E-04 0.05794  2.22747E-04 0.09475 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  7.55234E-01 0.05028  1.24902E-02 1.5E-05  3.17015E-02 0.00134  1.09293E-01 0.00035  3.17106E-01 0.00019  1.35342E+00 0.00031  8.61470E+00 0.00539 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 15 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  2.75316E-04 0.00618  2.75403E-04 0.00622  2.57480E-04 0.06626 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  3.39457E-04 0.00610  3.39564E-04 0.00614  3.17400E-04 0.06598 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  6.21459E-03 0.02808  2.13072E-04 0.13776  1.02088E-03 0.07729  1.00974E-03 0.07711  2.83616E-03 0.04299  8.67462E-04 0.08038  2.67267E-04 0.14946 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  7.26625E-01 0.07572  1.24902E-02 1.8E-05  3.17009E-02 0.00189  1.09228E-01 0.00071  3.17047E-01 0.00014  1.35360E+00 0.00025  8.54863E+00 0.01243 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  2.61033E-04 0.01339  2.61309E-04 0.01360  2.01908E-04 0.19937 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  3.21758E-04 0.01279  3.22099E-04 0.01301  2.48474E-04 0.19770 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  6.12391E-03 0.11285  1.49825E-04 0.52864  9.43298E-04 0.25719  1.59109E-03 0.24781  2.29272E-03 0.13418  9.66209E-04 0.21944  1.80764E-04 0.39055 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  6.49766E-01 0.20658  1.24906E-02 8.6E-09  3.17753E-02 0.00153  1.09372E-01 3.1E-05  3.16996E-01 2.1E-05  1.35325E+00 0.00050  8.63638E+00 0.0E+00 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  2.64281E-04 0.00315 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  3.25834E-04 0.00242 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  6.36706E-03 0.02195 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -2.40712E+01 0.02050 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  9.86017E-07 0.00194 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  3.40943E-05 0.00066  3.40916E-05 0.00065  3.45300E-05 0.00834 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.01895E-04 0.00342  6.02201E-04 0.00337  5.46824E-04 0.05990 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  6.68785E-01 0.00118  6.68020E-01 0.00119  8.34745E-01 0.02416 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.14649E+01 0.05336 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  1.61528E+02 0.00167  1.51978E+02 0.00208 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1:  17]) = 'Serpent pre-2.2.1' ;
COMPILE_DATE              (idx, [1:  20]) = 'Jul 11 2024 22:55:21' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:   8]) = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1:  28]) = 'gFHR_equilibrium_100.serpent' ;
WORKING_DIRECTORY         (idx, [1:  66]) = '/global/scratch/users/ikolaja/PEARLSim_dev/PEARLSim/gFHR_benchmark' ;
HOSTNAME                  (idx, [1:  12]) = 'n0024.savio4' ;
CPU_TYPE                  (idx, [1:  40]) = 'Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz' ;
CPU_MHZ                   (idx, 1)        = 218104675.0 ;
START_DATE                (idx, [1:  24]) = 'Sat Nov 15 03:56:21 2025' ;
COMPLETE_DATE             (idx, [1:  24]) = 'Sat Nov 15 04:02:39 2025' ;

% Run parameters:

POP                       (idx, 1)        = 10000 ;
CYCLES                    (idx, 1)        = 35 ;
SKIP                      (idx, 1)        = 15 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 42 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000 ;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ] ;
B1_IMPLICIT_LEAKAGE       (idx, 1)        = 0 ;
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 0 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 2 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 0 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 1 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 56 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_HISTORY_PROFILE       (idx, [1:  56]) = [  1.04983E+00  1.00448E+00  1.03381E+00  1.02318E+00  1.03258E+00  1.01657E+00  1.01478E+00  9.87571E-01  9.76039E-01  9.69320E-01  1.02452E+00  9.80741E-01  1.02844E+00  9.67865E-01  9.57452E-01  9.68760E-01  9.86228E-01  1.03325E+00  1.02407E+00  1.01456E+00  1.02172E+00  1.02642E+00  1.01545E+00  1.02049E+00  1.01859E+00  1.01108E+00  1.01702E+00  1.02508E+00  9.85444E-01  9.78838E-01  9.81301E-01  1.02015E+00  1.02519E+00  9.70328E-01  9.72455E-01  9.57228E-01  9.75703E-01  1.00784E+00  1.01478E+00  9.61482E-01  1.02732E+00  1.03270E+00  1.02743E+00  9.69768E-01  9.86900E-01  9.66633E-01  1.02015E+00  9.78950E-01  9.71000E-01  9.92162E-01  9.82533E-01  9.77942E-01  1.03146E+00  9.77046E-01  9.76934E-01  9.80405E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 0 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1:  71]) = '/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1:  67]) = '/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1:   3]) = 'N/A' ;
NFY_DATA_FILE_PATH        (idx, [1:  67]) = '/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:   3]) = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:   2]) = [  9.00000E-01  9.00000E-01 ] ;
ST_FRAC                   (idx, [1:   4]) = [  8.73876E-01 0.00016  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  1.26124E-01 0.00108  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  2.64469E-01 0.00025  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  9.92691E-01 2.3E-05  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  5.11077E-01 0.00086  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  3.28143E+01 0.00168  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
CELL_SEARCH_FRAC          (idx, [1:  10]) = [  9.82179E-01 7.9E-06  1.76995E-02 0.00043  1.21747E-04 0.00160  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  1.62951E+02 0.00162  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  1.62934E+02 0.00162  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  1.53535E+02 0.00049  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.33864E+03 0.00145  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 35 ;
SIMULATED_HISTORIES       (idx, 1)        = 349990 ;
MEAN_POP_SIZE             (idx, [1:   2]) = [  9.99971E+03 0.00304 ] ;
MEAN_POP_WGT              (idx, [1:   2]) = [  9.99971E+03 0.00304 ] ;
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  8.11925E+01 ;
RUNNING_TIME              (idx, 1)        =  6.30303E+00 ;
INIT_TIME                 (idx, [1:   2]) = [  2.10402E+00  2.10402E+00 ] ;
PROCESS_TIME              (idx, [1:   2]) = [  5.08500E-02  7.90000E-03 ] ;
TRANSPORT_CYCLE_TIME      (idx, [1:   3]) = [  3.86227E+00  1.15327E+00  1.22903E+00 ] ;
BURNUP_CYCLE_TIME         (idx, [1:   2]) = [  2.01000E-02  8.93333E-03 ] ;
BATEMAN_SOLUTION_TIME     (idx, [1:   2]) = [  1.57167E-02  1.10000E-03 ] ;
MPI_OVERHEAD_TIME         (idx, [1:   2]) = [  0.00000E+00  0.00000E+00 ] ;
ESTIMATED_RUNNING_TIME    (idx, [1:   2]) = [  6.30217E+00  6.30217E+00 ] ;
CPU_USAGE                 (idx, 1)        = 12.88150 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  5.38293E+01 0.02789 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  2.13188E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 515428.37 ;
ALLOC_MEMSIZE             (idx, 1)        = 47337.82 ;
MEMSIZE                   (idx, 1)        = 46829.59 ;
XS_MEMSIZE                (idx, 1)        = 20248.30 ;
MAT_MEMSIZE               (idx, 1)        = 893.17 ;
RES_MEMSIZE               (idx, 1)        = 2.61 ;
IFC_MEMSIZE               (idx, 1)        = 0.00 ;
MISC_MEMSIZE              (idx, 1)        = 25685.51 ;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00 ;
UNUSED_MEMSIZE            (idx, 1)        = 508.24 ;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 1309 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  5.00000E-05 ;
NEUTRON_ERG_NE            (idx, 1)        = 272467 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-11 ;
NEUTRON_EMAX              (idx, 1)        =  2.00000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.10000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 232 ;
URES_USED                 (idx, 1)        = 115 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 2655 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 1520 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1135 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 34258 ;
TOT_TRANSMU_REA           (idx, 1)        = 1727 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:   9]) = [ 0 0 0 0 0 0 0 0 0 ] ;
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  5.26843E+19 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.72853E+07 ;
TOT_SF_RATE               (idx, 1)        =  8.88883E+05 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  6.17988E+18 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  4.32517E+05 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.65044E+19 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.68528E+07 ;
INHALATION_TOXICITY       (idx, 1)        =  3.06283E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  2.30883E+10 ;
ACTINIDE_INH_TOX          (idx, 1)        =  6.37668E+09 ;
ACTINIDE_ING_TOX          (idx, 1)        =  2.56930E+09 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.42516E+10 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.05190E+10 ;
SR90_ACTIVITY             (idx, 1)        =  2.12274E+15 ;
TE132_ACTIVITY            (idx, 1)        =  3.76507E+17 ;
I131_ACTIVITY             (idx, 1)        =  2.51598E+17 ;
I132_ACTIVITY             (idx, 1)        =  3.80970E+17 ;
CS134_ACTIVITY            (idx, 1)        =  5.33193E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.26421E+15 ;
PHOTON_DECAY_SOURCE       (idx, [1:   2]) = [  4.48478E+19  8.13617E+06 ] ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  9.14563E+16 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.88509E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  6.33736E+19 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  1.74182E+15 0.00183  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 320 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:   3]) = [ 0 0 0 ] ;
BURNUP                    (idx, [1:   2]) = [  2.20653E+00  2.20719E+00 ] ;
BURN_DAYS                 (idx, [1:   2]) = [  6.52500E+00  6.52500E+00 ] ;
FIMA                      (idx, [1:   3]) = [  2.31497E-03  4.85984E+24  2.09445E+27 ] ;

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  2.93306E-01 0.00507 ];
U235_FISS                 (idx, [1:   4]) = [  7.70035E+18 0.00115  8.93886E-01 0.00113 ];
U238_FISS                 (idx, [1:   4]) = [  1.73400E+16 0.05360  2.01290E-03 0.05360 ];
PU239_FISS                (idx, [1:   4]) = [  8.73085E+17 0.00963  1.01352E-01 0.00966 ];
PU240_FISS                (idx, [1:   4]) = [  1.99875E+14 0.47753  2.32026E-05 0.47753 ];
PU241_FISS                (idx, [1:   4]) = [  2.24874E+16 0.05247  2.61047E-03 0.05248 ];
U235_CAPT                 (idx, [1:   4]) = [  1.88985E+18 0.00625  2.18606E-01 0.00572 ];
U238_CAPT                 (idx, [1:   4]) = [  3.05524E+18 0.00507  3.53347E-01 0.00302 ];
PU239_CAPT                (idx, [1:   4]) = [  5.27188E+17 0.01091  6.09580E-02 0.00954 ];
PU240_CAPT                (idx, [1:   4]) = [  1.77421E+17 0.01412  2.05249E-02 0.01403 ];
PU241_CAPT                (idx, [1:   4]) = [  8.81686E+15 0.07438  1.02001E-03 0.07409 ];
XE135_CAPT                (idx, [1:   4]) = [  4.55939E+17 0.01016  5.27296E-02 0.00922 ];
XE135M_CAPT               (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC      (idx, [1:   2]) = [ 0 0.00000E+00 ] ;
BALA_SRC_NEUTRON_FISS     (idx, [1:   2]) = [ 349990 3.50000E+05 ] ;
BALA_SRC_NEUTRON_NXN      (idx, [1:   2]) = [ 0 2.80870E+03 ] ;
BALA_SRC_NEUTRON_VR       (idx, [1:   2]) = [ 0 0.00000E+00 ] ;
BALA_SRC_NEUTRON_TOT      (idx, [1:   2]) = [ 349990 3.52809E+05 ] ;

BALA_LOSS_NEUTRON_CAPT    (idx, [1:   2]) = [ 172340 1.73714E+05 ] ;
BALA_LOSS_NEUTRON_FISS    (idx, [1:   2]) = [ 171711 1.73118E+05 ] ;
BALA_LOSS_NEUTRON_LEAK    (idx, [1:   2]) = [ 5939 5.97655E+03 ] ;
BALA_LOSS_NEUTRON_CUT     (idx, [1:   2]) = [ 0 0.00000E+00 ] ;
BALA_LOSS_NEUTRON_ERR     (idx, [1:   2]) = [ 0 0.00000E+00 ] ;
BALA_LOSS_NEUTRON_TOT     (idx, [1:   2]) = [ 349990 3.52809E+05 ] ;

BALA_NEUTRON_DIFF         (idx, [1:   2]) = [ 0 -3.20142E-09 ] ;

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  2.80000E+08 0.0E+00  2.80000E+08 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.38165E-01 0.0E+00  3.38165E-01 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.13825E+19 0.00015  2.13825E+19 0.00015  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  8.61445E+18 2.8E-05  8.61445E+18 2.8E-05  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  8.64604E+18 0.00363  6.52396E+18 0.00392  2.12208E+18 0.00609 ];
TOT_ABSRATE               (idx, [1:   6]) = [  1.72605E+19 0.00182  1.51384E+19 0.00169  2.12208E+18 0.00609 ];
TOT_SRCRATE               (idx, [1:   6]) = [  1.74182E+19 0.00183  1.74182E+19 0.00183  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  7.56670E+21 0.00240  3.22666E+19 0.00189  7.53443E+21 0.00241 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.97468E+17 0.01339 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  1.75580E+19 0.00184 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  2.83813E+21 0.00256 ];
INI_FMASS                 (idx, 1)        =  8.27997E+02 ;
TOT_FMASS                 (idx, 1)        =  8.26102E+02 ;
INI_BURN_FMASS            (idx, 1)        =  8.27997E+02 ;
TOT_BURN_FMASS            (idx, 1)        =  8.26102E+02 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.84428E+00 0.00151 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.39213E-01 0.00103 ];
SIX_FF_P                  (idx, [1:   2]) = [  6.62832E-01 0.00096 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.21759E+00 0.00107 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.99163E-01 4.9E-05 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.83747E-01 0.00023 ];
SIX_FF_KINF               (idx, [1:   2]) = [  1.24907E+00 0.00182 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  1.22774E+00 0.00184 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.48217E+00 0.00017 ];
FISSE                     (idx, [1:   2]) = [  2.02871E+02 2.8E-05 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.22721E+00 0.00207  1.22014E+00 0.00189  7.60051E-03 0.03244 ];
COL_KEFF                  (idx, [1:   2]) = [  1.22774E+00 0.00184 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 2.000000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  1.79484E+01 0.00031 ];
IMP_ALF                   (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  3.20881E-07 0.00552 ];
IMP_EALF                  (idx, [1:   2]) = [  2.00000E+01 0.0E+00 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  1.00554E-02 0.03549 ];
IMP_AFGE                  (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  5.06040E-03 0.02576  1.91222E-04 0.12742  8.56939E-04 0.03952  7.96504E-04 0.06858  2.33268E-03 0.03710  6.57381E-04 0.04773  2.25670E-04 0.09824 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  7.38292E-01 0.05261  1.24901E-02 1.9E-05  3.17125E-02 0.00080  1.09317E-01 0.00035  3.17181E-01 0.00023  1.35119E+00 0.00146  8.68396E+00 0.00306 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 15 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  2.77835E-04 0.00637  2.77738E-04 0.00628  2.85848E-04 0.07177 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  3.40912E-04 0.00601  3.40794E-04 0.00593  3.50582E-04 0.07153 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  6.22069E-03 0.03400  1.94821E-04 0.18491  1.05334E-03 0.05755  1.00681E-03 0.09184  2.85487E-03 0.05033  8.24088E-04 0.06540  2.86762E-04 0.14039 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  7.46192E-01 0.07010  1.24899E-02 4.0E-05  3.17300E-02 0.00100  1.09411E-01 0.00076  3.17070E-01 0.00017  1.35282E+00 0.00036  8.63638E+00 0.0E+00 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  2.64225E-04 0.01070  2.64084E-04 0.01046  2.78231E-04 0.17338 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  3.24165E-04 0.01009  3.23997E-04 0.00989  3.41249E-04 0.17266 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  6.07836E-03 0.10078  2.42115E-04 0.43810  8.80741E-04 0.22777  1.05327E-03 0.23528  2.83238E-03 0.14653  7.58937E-04 0.26804  3.10926E-04 0.30767 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  7.35056E-01 0.17007  1.24890E-02 0.00013  3.16624E-02 0.00356  1.09164E-01 0.00133  3.17063E-01 8.9E-05  1.32552E+00 0.02147  8.63638E+00 0.0E+00 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  2.71354E-04 0.00341 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  3.32961E-04 0.00273 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  6.09922E-03 0.01210 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -2.24758E+01 0.01157 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  9.86566E-07 0.00197 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  3.40479E-05 0.00059  3.40457E-05 0.00058  3.44261E-05 0.00749 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.01355E-04 0.00347  6.01706E-04 0.00350  5.41469E-04 0.04387 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  6.70149E-01 0.00093  6.69481E-01 0.00097  8.26076E-01 0.04526 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.12637E+01 0.04122 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  1.61744E+02 0.00163  1.52369E+02 0.00222 ];

