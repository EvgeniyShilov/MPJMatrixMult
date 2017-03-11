import mpi.Intracomm;
import mpi.MPI;

import java.io.IOException;
import java.util.Date;
import java.util.Random;

public class Node {

    private static final int DEFAULT_L = 4;
    private static final int DEFAULT_M = 5;
    private static final int DEFAULT_N = 6;
    private static final int DEFAULT_RANDOM_RANGE = 10;

    private static void print(int[] matrix, int rowCount, int columnCount) {
        System.out.println("Matrix:");
        for (int i = 0; i < rowCount; i++) {
            for (int j = 0; j < columnCount; j++)
                System.out.print(matrix[i * columnCount + j] + "\t");
            System.out.println();
        }
    }

    private static void init(int[] matrix, int rowCount, int columnCount) {
        Random random = new Random();
        for (int i = 0; i < rowCount; i++)
            for (int j = 0; j < columnCount; j++)
                matrix[i * columnCount + j] = random.nextInt(DEFAULT_RANDOM_RANGE);
    }

    public static void main(String[] args) throws IOException {
        MPI.Init(args);
        boolean output = true;
        Intracomm intracomm;
        if(args.length < 7) intracomm = MPI.COMM_WORLD;
        else {
            output = false;
            int rank = MPI.COMM_WORLD.Rank();
            intracomm = MPI.COMM_WORLD.Split(rank % Integer.parseInt(args[6]), rank);
        }
        int rank = intracomm.Rank();
        int size = intracomm.Size();
        int[] LMN = new int[3];
        if(rank == 0) {
            LMN[0] = args.length < 6 ? DEFAULT_L : Integer.parseInt(args[3]);
            LMN[1] = args.length < 6 ? DEFAULT_M : Integer.parseInt(args[4]);
            LMN[2] = args.length < 6 ? DEFAULT_N : Integer.parseInt(args[5]);
            for (int i = 1; i < size; i++)
                intracomm.Send(LMN, 0, 3, MPI.INT, i, 0);
        }
        else intracomm.Recv(LMN, 0, 3, MPI.INT, 0, 0);
        int L = LMN[0];
        int M = LMN[1];
        int N = LMN[2];
        int[] A = new int[L * M];
        int[] B = new int[M * N];
        int[] C = new int[L * N];
        if(rank == 0) {
            init(A, L, M);
            init(B, M, N);
            if(output) {
                print(A, L, M);
                print(B, M, N);
            }
            System.out.println("Start multiplying");
        }
        double begin = new Date().getTime();
        intracomm.Bcast(B, 0, M * N, MPI.INT, 0);
        int linesPerNode = L / size;
        if(linesPerNode != 0) {
            int[] linesA = new int[M * linesPerNode];
            intracomm.Scatter(A, 0, M * linesPerNode, MPI.INT, linesA, 0, M * linesPerNode, MPI.INT, 0);
            int[] linesC = new int[N * linesPerNode];
            for (int i = 0; i < linesPerNode; i++)
                for (int j = 0; j < N; j++)
                    for (int r = 0; r < M; r++)
                        linesC[i * N + j] += linesA[i * M + r] * B[r * N + j];
            intracomm.Gather(linesC, 0, N * linesPerNode, MPI.INT, C, 0, N * linesPerNode, MPI.INT, 0);
        }
        int lastLines = L % size;
        for (int i = L - lastLines; i < L; i++)
            for (int j = 0; j < N; j++)
                for (int r = 0; r < M; r++)
                    C[i * N + j] += A[i * M + r] * B[r * N + j];
        double end = new Date().getTime();
        if(rank == 0) {
            System.out.println("Finish multiplying");
            if(output) print(C, L, N);
            System.out.println("Time: " + (end - begin));
        }
        MPI.Finalize();
    }
}