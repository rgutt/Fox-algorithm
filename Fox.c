#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include <math.h>


typedef struct{
    int p;
    MPI_Comm comm;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    int q;
    int my_row;
    int my_col;
    int my_rank;
}GRID_INFO_T;


void getDim(
	int dimM[],
	FILE* fp
	)
{
    	char * line = NULL;
    	size_t len = 0;
	getline(&line, &len, fp);
	
	
	
	dimM[0] = atoi(line);
	getline(&line, &len, fp);
	dimM[1] = atoi(line);
}



void Setup_grid(
    GRID_INFO_T* grid
)
{		
    int wrap_around[2];
    int dimensions[2];
    int coordinates[2];
    int free_coords[2];
    int old_rank;

    MPI_Comm_size(MPI_COMM_WORLD,&(grid->p));
    MPI_Comm_rank(MPI_COMM_WORLD,&old_rank);
    
    grid->q = (int) sqrt((double) grid->p);
    dimensions[0] = dimensions[1] = grid->q;
    
    wrap_around[0] = wrap_around[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD,2,dimensions,wrap_around,1,&(grid->comm));
    MPI_Comm_rank(grid->comm,&(grid->my_rank));
    
    MPI_Cart_coords(grid->comm,grid->my_rank,2,coordinates);
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];
    
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(grid->comm, free_coords,&(grid->row_comm));
    
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(grid->comm, free_coords,&(grid->col_comm));
}


void readMatFromFile(
	double M_loc[],
	int n_loc,
        int row,
	FILE* fp
  
)
{
	int i;
	char * line = NULL;
    	size_t len = 0;
	
	for (i = 0; i < n_loc;i++)
	{
	    getline(&line, &len, fp);
	    /*Converting string to double*/
	    M_loc[i+row*n_loc] = strtod(line, NULL);
	}
}

void getMatrix(
	double M_loc[],
	int dimM[],
        int m_loc,
	int n_loc,
	GRID_INFO_T grid,
	FILE* fp,
	int source
     )
{
	double* temp;
        int dest;
	int row,col,i,q;
    	char * line = NULL;
    	size_t len = 0;
	MPI_Status status;
	
        
        temp = malloc(n_loc*sizeof(double));   
                  
        for (row = 0; row < dimM[0]; row ++)
        {
            for ( col = 0; col < dimM[1]; col += n_loc)
            {
                if (grid.my_rank == source)
                {
                    if ( grid.my_row == row/m_loc && grid.my_col == col/n_loc)
                    {
                        for (i = 0; i < n_loc;i++)
                        {
                            getline(&line, &len, fp);
                            /*Converting string to double*/
                            M_loc[i+row*n_loc] = strtod(line, NULL);
                        }
                    }
                    else
                    {
                        for (i = 0; i < n_loc;i++)
                        {
                            getline(&line, &len, fp);
                            /*Converting string to double*/
                            temp[i] = strtod(line, NULL);
                        }
                        dest = (row/m_loc)*grid.q+col/n_loc;
                        MPI_Send(temp,n_loc,MPI_DOUBLE,dest,0,MPI_COMM_WORLD); 
                        
                    }
                }
                else
                {
                    dest = (row/m_loc)*grid.q+col/n_loc;
                    if (grid.my_rank == dest)
                    {
                        MPI_Recv(temp,n_loc,MPI_DOUBLE,source,0,MPI_COMM_WORLD,&status);
                        for (i = 0; i < n_loc;i++)
                        {
                            M_loc[i+n_loc*(row-m_loc*grid.my_row)] = temp[i];
                        }
                        
                    }
                }
            }
        }	
        free(temp);
	
}


void locMatmul(
	double A_loc[],
	double B_loc[],
	double C_loc[],
	int m_loc,
	int n_loc,
	int l_loc,
	GRID_INFO_T grid
)
{
	int i,j,k;
	
	/*Outer loop over Rows of local A, second loop over columns of local A(n_loc) and third loop over the columns of local B(l)*/
	for (k = 0; k < m_loc; k++)
	{
		for ( i = 0; i < n_loc; i++)
		{
 			for ( j = 0; j < l_loc; j++)
			{
			    C_loc[j+k*l_loc] = C_loc[j+k*l_loc] + A_loc[i+k*n_loc]*B_loc[j+i*l_loc];
			}
		}
	}
	
}



void matmul(
        double A_loc[],
	double B_loc[],
	double C_loc[],
	int m_loc,		//A_loc is m_locxn_loc
	int n_loc,		//B_loc is n_locxl_loc
	int l_loc,		//C_loc is m_locxl_loc
	GRID_INFO_T grid
	)
{
	double* temp;
	int i,j,stage,k_bar,k;
	int dest, source;
	MPI_Status status;
	temp = malloc(n_loc*m_loc*sizeof(double));
	memset(C_loc, 0, m_loc*l_loc * sizeof(double));		//Set all values of C_loc to zero
        i = grid.my_row;
        j = grid.my_col;
        dest = ((i+grid.q-1)%grid.q);
        source = (i+1)%grid.q;
        for ( stage = 0; stage < grid.q; stage ++)
        {
            k_bar = (stage+i)%grid.q;		
            if (grid.my_col == k_bar)
            {
                memcpy(temp,A_loc, m_loc*n_loc*sizeof(double));
            }
            
	    MPI_Bcast(temp,m_loc*n_loc,MPI_DOUBLE,k_bar,grid.row_comm); 
	    locMatmul( temp,B_loc, C_loc,m_loc,n_loc,l_loc,grid );
	    MPI_Sendrecv_replace(B_loc,n_loc*l_loc, MPI_DOUBLE, dest, 0, source, 0, grid.col_comm, &status);   
        }		

}

void writeCtotxt(
      double C_loc[],
      int dimC[],
      int m_loc,
      int l_loc,
      int source,
      GRID_INFO_T grid
     )
{
    int row,col,i;
    int src;
    double* temp;
    FILE* fpC;
    MPI_Status status;
    if ( grid.my_rank == source)
    {
	fpC = fopen("C.txt", "w+");
        fprintf(fpC,"%i\n",dimC[0]);
        fprintf(fpC,"%i\n",dimC[1]);
    }
    for ( row = 0; row < dimC[0];row ++)
    {
      for ( col = 0; col < dimC[1];col+=l_loc)
      {    
	if (grid.my_rank == source)
	{
	  temp = malloc(l_loc * sizeof(double));  
	  if ( grid.my_row == row/m_loc && grid.my_col == col/l_loc)
	  {
	    for ( i = 0; i < l_loc;i++)        
	    {
	      fprintf(fpC,"%lf\n",C_loc[i+row*l_loc]);
	    }
	  }
	  else
	  {
	    src = (row/m_loc)*grid.q+col/l_loc;
	    MPI_Recv(temp,l_loc,MPI_DOUBLE,src,0,MPI_COMM_WORLD,&status);
	    for ( i = 0; i < l_loc;i++)
	    {
	      fprintf(fpC,"%lf\n",temp[i]);
	    }
	  }
	}
	else
	{
	  src = (row/m_loc)*grid.q+col/l_loc;
	  if (grid.my_rank == src )
	  {
	      MPI_Send(&C_loc[l_loc*(row-m_loc*grid.my_row)],l_loc,MPI_DOUBLE,source,0,MPI_COMM_WORLD);
	  }
	}
	
      }

    }
    if ( grid.my_rank == source)
    {
	fclose(fpC);
	free(temp);
    }
  
}

int main(int argc,char* argv[])
{
	int* dimA;
	int* dimB;
	int* dimC;
        int m_loc;
	int n_loc;
        int l_loc;
	int i;
	int source = 0;
	double* A_loc;
	double* B_loc;
	double* C_loc;
        double start_tot,stop_tot,loc_time,final_time;
        double start_fox,stop_fox,loc_time_fox,final_time_fox;
        double norm_loc,norm_glob;
	FILE* fpA;
	FILE* fpB;
	FILE* fpTime;
	MPI_Init(&argc, &argv);
	GRID_INFO_T grid;

	MPI_Status status;
	
        start_tot = MPI_Wtime();
        if (grid.my_rank == 3)
        {
            for (i = 0; i < l_loc*m_loc;i++)
            {
                    if (i%l_loc == 0)
                    {
                        printf("\n");
                    }
                    printf("%lf ",C_loc[i]);

            }
            printf("\n");
        }
        
        Setup_grid(&grid);
	dimA = malloc(2*sizeof(int));
	dimB = malloc(2*sizeof(int));
	dimC = malloc(2*sizeof(int));
	
	if (grid.my_rank == source)
	{
		fpA = fopen("A.txt", "r");        if (grid.my_rank == 3)
        {
            for (i = 0; i < l_loc*m_loc;i++)
            {
                    if (i%l_loc == 0)
                    {
                        printf("\n");
                    }
                    printf("%lf ",C_loc[i]);

            }
            printf("\n");
        }
		fpB = fopen("B.txt", "r");
		/*Read dimensions of A and B from Files*/
		getDim(dimA,fpA);
		getDim(dimB,fpB);
	}
	
	/*Broadcasting dimensions of matrix */
	MPI_Bcast(dimA,2,MPI_INT,source,MPI_COMM_WORLD);
	MPI_Bcast(dimB,2,MPI_INT,source,MPI_COMM_WORLD);
	
	dimC[0] = dimA[0];
	dimC[1] = dimB[1];
	
        m_loc = dimA[0]/grid.q;
	n_loc = dimA[1]/grid.q;
        l_loc = dimB[1]/grid.q;
//  	printf("%i \n\n",n_loc);
	A_loc = malloc(n_loc*m_loc*sizeof(double));
	B_loc = malloc(n_loc*l_loc*sizeof(double));
	C_loc = malloc(m_loc*l_loc*sizeof(double));
	
	/*Initial local Matrices*/

	getMatrix( A_loc,dimA,m_loc, n_loc, grid, fpA ,source);
   	getMatrix( B_loc,dimB,n_loc, l_loc, grid, fpB ,source);
        start_fox = MPI_Wtime();
  	matmul(A_loc, B_loc, C_loc,m_loc,n_loc,l_loc, grid);
        
        stop_fox = MPI_Wtime();
	
        writeCtotxt( C_loc,dimC, m_loc, l_loc, source,grid );
        
        stop_tot = MPI_Wtime();
        
        
//         norm_loc = getNorm(C_loc,m_loc,l_loc);
        
        loc_time = stop_tot-start_tot;
        loc_time_fox = stop_fox-start_fox;
        MPI_Reduce(&loc_time, &final_time, 1, MPI_DOUBLE, MPI_MAX, source,MPI_COMM_WORLD);
        MPI_Reduce(&loc_time_fox, &final_time_fox, 1, MPI_DOUBLE, MPI_MAX, source,MPI_COMM_WORLD);
        if ( grid.my_rank == source){
                fpTime = fopen("time.txt","a");
                fprintf(fpTime,"%i\t%lf\t%lf\n",grid.p,final_time,final_time_fox);
                fclose(fpTime);
	}
        
	if (grid.my_rank == source)
	{
		  fclose(fpA);
		  fclose(fpB);
	}

	
	
	free(A_loc);free(B_loc);free(C_loc);
	
	MPI_Finalize();

	return 0;
}
