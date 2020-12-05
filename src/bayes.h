int  DoQuit (void);
void GetTimeSeed (void);
void InitializeMrBayes (void);
void MrBayesPrint (char *, ...);
void MrBayesPrintf (FILE *f, char *format, ...);
int  ReinitializeMrBayes (void);
void SetCode (int part);
void SetModelDefaults (void);

extern int InitializeLinks (void);
int CommandLine (int argc, char **argv);
int			defTaxa;                     /* flag for whether number of taxa is known      */
int			defChars;                    /* flag for whether number of characters is known*/
int			defMatrix;                   /* flag for whether matrix is successfull read   */
int			defPartition;                /* flag for whether character partition is read  */
int			defConstraints;              /* flag for whether constraints on tree are read */
int			defPairs;                    /* flag for whether pairs are read               */
Doublet		doublet[16];                 /* holds information on states for doublets      */
int			fileNameChanged;			 /* has file name been changed ?                  */
long int	globalSeed;                  /* seed that is initialized at start up          */
int			nBitsInALong;                /* number of bits in a long                      */
int			readWord;					 /* should we read word next ?                    */
long int	runIDSeed;                   /* seed used only for determining run ID [stamp] */
long int	swapSeed;                    /* seed used only for determining which to swap  */
int         userLevel;                   /* user level                                    */
#			if defined (MPI_ENABLED)
int 		proc_id;                     /* process ID (0, 1, ..., num_procs-1)           */
int 		num_procs;                   /* number of active processors                   */
MrBFlt		myStateInfo[4];              /* likelihood/prior/heat vals of me              */
MrBFlt		partnerStateInfo[4];		 /* likelihood/prior/heat vals of partner         */
#			endif