package vcu.edu.datastreamlearning.ollawv;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import com.yahoo.labs.samoa.instances.Instances;

public class Logger extends PrintWriter {
	
	protected PrintWriter logger;
	
	/* Constructor to System.out output */
	Logger(){
		super(System.out,true);
		logger = new PrintWriter(System.out, true);
	}
	
	/* Constructor to file output */
	Logger(String filename) throws FileNotFoundException{
		super(filename);
		try {
			logger = new PrintWriter(new FileWriter(filename), true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/* Print barrier */
	public void printBarrier(){
		logger.printf("-----------------------------------------\n");
	}
	
	/* Prints chunk of (Instances) data */
	public void printInstances(Instances chunk){
		for(int i = 0; i < chunk.size(); i++){
			double label = chunk.get(i).classValue();
			logger.print(label+" ");
			for(int j = 1; j < chunk.numAttributes(); j++){
				logger.print(j+":"+chunk.get(i).value(j-1)+" ");
			}
			logger.println();
		}
	}
}
