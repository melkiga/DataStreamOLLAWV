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
				"-l \"vcu.edu.datastreamlearning.ollawv.OLLASolver -c 1.0 -g 2.0\"", 
				"-i "+10000, 
				"-f "+100, 
				"-c "+100,
				"-d results/output.csv"};
		
		/* Calls Evaluate Interleaved Chunks */
		DoTask.main(localArgs);
		
		log.close();
	}

}
