package vcu.edu.datastreamlearning.ollawv;

import moa.DoTask;


public class OLLAWVTester {
	
	public static void main(String[] args){
		
		Logger log = new Logger();
		/* Set up data configuration. */
		int numChunks = 5;
		int chunkSize = 20;
		int dim = 2;
		int seed = 1;
		int numClasses = 2;
		int sampleFreq = chunkSize;
//		String[] localArgs = {"moa.tasks.EvaluateInterleavedChunksG -S ", 
//				"-s \"generators.RandomRBFGenerator -r "+seed+" -i "+seed+" -c "+numClasses+" -a "+dim+"\"", 
//				"-l \"vcu.edu.datastreamlearning.ollawv.OLLASolver -c 1.0 -g 4.5 -v 1\"", 
//				"-i "+chunkSize*numChunks, 
//				"-f "+sampleFreq, 
//				"-c "+chunkSize};
		
		String[] localArgs = {"moa.tasks.EvaluateInterleavedChunksG -S ", 
				"-s \"generators.RandomRBFGenerator\"",
				"-l \"vcu.edu.datastreamlearning.ollawv.OLLASolver -s 1 -p 1 -v 1\"", 
				"-i "+1000000, 
				"-f "+1000, 
				"-c "+1000,
				"-d results/output.csv"};
		
		//String[] localArgs = {"moa.tasks.EvaluateInterleavedChunksG -s ",
		//			"\"ConceptDriftStream -s (generators.RandomTreeGenerator -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 2 -r 2 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 3 -r 3 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 7 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 11 -r 2 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 5 -r 3 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 115 -r 1 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 18 -r 2 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 25 -r 3 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 61 -r 1 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 71 -r 2 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 31 -r 3 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 17 -r 1 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 64 -r 2 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 66 -r 3 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 21 -r 1 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 88 -r 2 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 55 -r 3 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 99 -r 1 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 74 -r 2 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 57 -r 3 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 1 -r 1 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 6 -r 2 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 3 -r 3 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 86 -r 1 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 49 -r 2 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 69 -r 3 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 96 -r 1 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 81 -r 2 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 41 -r 3 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 17 -r 1 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 71 -r 2 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 7 -r 3 -c 6) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -i 13 -r 1 -c 6) -d (generators.RandomTreeGenerator -i 29 -r 2 -c 6) -a 90.0 -p 3000 -w 1 -r 63) -a 90.0 -p 3000 -w 1 -r 63) -a 90.0 -p 3000 -w 1 -r 63) -a 90.0 -p 3000 -w 1 -r 63) -a 90.0 -p 3000 -w 1 -r 33) -a 90.0 -p 3000 -w 1 -r 23) -a 90.0 -p 3000 -w 1 -r 32) -a 90.0 -p 3000 -w 1 -r 4) -a 90.0 -p 3000 -w 1 -r 7) -a 90.0 -p 3000 -w 1 -r 37) -a 90.0 -p 3000 -w 1 -r 73) -a 90.0 -p 3000 -w 1 -r 36) -a 90.0 -p 3000 -w 1 -r 89) -a 90.0 -p 3000 -w 1 -r 99) -a 90.0 -p 3000 -w 1 -r 97) -a 90.0 -p 3000 -w 1 -r 73) -a 90.0 -p 3000 -w 1 -r 43) -a 90.0 -p 3000 -w 1 -r 34) -a 90.0 -p 3000 -w 1 -r 38) -a 90.0 -p 3000 -w 1 -r 32) -a 90.0 -p 3000 -w 1 -r 3) -a 90.0 -p 3000 -w 1 -r 33) -a 90.0 -p 3000 -w 1 -r 77) -a 90.0 -p 3000 -w 1 -r 69) -a 90.0 -p 3000 -w 1 -r 96) -a 90.0 -p 3000 -w 1 -r 29) -a 90.0 -p 3000 -w 1 -r 36) -a 90.0 -p 3000 -w 1 -r 77) -a 90.0 -p 3000 -w 1 -r 37) -a 90.0 -p 3000 -w 1 -r 3) -a 90.0 -p 3000 -w 1 -r 311) -a 90.0 -p 3000 -w 1 -r 23) -a 90.0 -p 3000 -w 1 -r 2) -a 90.0 -p 3000 -w 1\" -l \"vcu.edu.datastreamlearning.ollawv.OLLASolver -s 0\" ",
		//			"-i 10000 -f 100 -c 100"};
		
		/* Calls Evaluate Interleaved Chunks */
		DoTask.main(localArgs);
		
		log.close();
	}

}
