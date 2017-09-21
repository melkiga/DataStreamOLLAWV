package vcu.edu.datastreamlearning.ollawv;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import com.yahoo.labs.samoa.instances.Instances;

public class Logger {
	
	protected PrintWriter logger;
	
	/* Constructor to System.out output */
	public Logger(){
		logger = new PrintWriter(System.out, true);
	}
	
	/* Constructor to file output */
	public Logger(String filename){
		try {
			logger = new PrintWriter(new FileWriter(filename), true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/* Prints string */
	public <T> void print(T words){
		logger.println(words);
		logger.flush();
	}
	
	/* Prints a formatted string with generic single argument */
	public <T> void printFormatted(String words, T arg){
		logger.printf(words, arg);
		logger.flush();
	}
	
	/* Print barrier */
	public void printBarrier(){
		logger.print("-----------------------------------------\n");
		logger.flush();
	}
	
	/* Prints chunk of (Instances) data */
	public void printInstances(Instances chunk){
		for(int i = 0; i < chunk.size(); i++){
			double label = chunk.get(i).classValue();
			if(label == 0){
				label= -1;
			}
			logger.print(label+" ");
			for(int j = 1; j <= chunk.numAttributes(); j++){
				logger.print(j+":"+chunk.get(i).value(j-1)+" ");
			}
			logger.println();
		}
		logger.flush();
	}
}
