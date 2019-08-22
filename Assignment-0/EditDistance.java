/*
 * ------------------------------------------------------
 * @Author: Kaustav Vats (kaustav16048@iiitd.ac.in)
 * @Roll-No: 2016048
 * @Date: Thursday August 15th 2019 1:55:43 am
 * ------------------------------------------------------
 */

import java.io.*;
import java.util.*;
import java.math.*;
import java.lang.*;


public class EditDistance {

    /**
     * @param s1 Word
     * @param s2 Word
     * @param ShowChanges Boolean to show changes on string s1 to s2.
     * @return Cost to change Word s1 to s2.
     */
    private int getEditCost(String s1, String s2, boolean ShowChanges) {
        char[] A = s1.toCharArray();
        char[] B = s2.toCharArray();

        int[][] DP = new int[A.length+1][B.length+1];
        for (int i=0; i<=A.length; i++) {
            for (int j=0; j<=B.length; j++) {
                
                if (i == 0) {
                    DP[i][j] = j;
                }
                else if (j == 0) {
                    DP[i][j] = i;
                }
                else if (A[i-1] == B[j-1]) {
                    DP[i][j] = DP[i-1][j-1];
                }
                else {
                    DP[i][j] = 1 + Math.min(
                        DP[i-1][j-1]+1, 
                        Math.min(
                            DP[i-1][j], 
                            DP[i][j-1]
                        )
                    );
                }
            }
        }
        if (ShowChanges) {
            System.out.println("\nEdits performed on the String:");
            int i=A.length, j=B.length;
            while(i > 0 || j > 0) {
                if (i>0 && j>0 && A[i-1] == B[j-1]) {
                    i--;
                    j--;
                }
                else if (i>0 && j>0 && DP[i][j] == (DP[i-1][j-1] + 2)) {
                    System.out.println("Replace: " + A[i-1] + " -> " + B[j-1]);
                    i--;
                    j--;
                }
                else if (i>0 && DP[i][j] == (DP[i-1][j] + 1)) {
                    System.out.println("Remove: " + A[i-1]);
                    i--;
                }
                else if (j>0 && DP[i][j] == (DP[i][j-1]+1)) {
                    System.out.println("Insert: " + B[j-1]);
                    j--;
                }
            }
        }
        return DP[A.length][B.length];
    }

    /**
     * @param s1 Tokenized Words
     * @param s2 Tokenized Words
     * @param ShowChanges Boolean to show changes on string s1 to s2.
     * @return Cost to change String s1 to s2.
     */
    private int getEditCost(String[] s1, String[] s2, boolean ShowChanges) {
        int[][] DP = new int[s1.length+1][s2.length+1];
        for (int i=0; i<=s1.length; i++) {
            for (int j=0; j<=s2.length; j++) {
                
                if (i == 0) {
                    DP[i][j] = j;
                }
                else if (j == 0) {
                    DP[i][j] = i;
                }
                else if (s1[i-1].compareTo(s2[j-1]) == 0) {
                    DP[i][j] = DP[i-1][j-1];
                }
                else {
                    DP[i][j] = 1 + Math.min(
                        DP[i-1][j-1]+1, 
                        Math.min(
                            DP[i-1][j], 
                            DP[i][j-1]
                        )
                    );
                }
            }
        }
        if (ShowChanges) {
            System.out.println("\nEdits performed on the String:");
            int i=s1.length, j=s2.length;
            while(i > 0 || j > 0) {
                if (i>0 && j>0 && s1[i-1].compareTo(s2[j-1]) == 0) {
                    i--;
                    j--;
                }
                else if (i>0 && j>0 && DP[i][j] == (DP[i-1][j-1] + 2)) {
                    System.out.println("Replace: " + s1[i-1] + " -> " + s2[j-1]);
                    i--;
                    j--;
                }
                else if (i>0 && DP[i][j] == (DP[i-1][j] + 1)) {
                    System.out.println("Remove: " + s1[i-1]);
                    i--;
                }
                else if (j>0 && DP[i][j] == (DP[i][j-1]+1)) {
                    System.out.println("Insert: " + s2[j-1]);
                    j--;
                }
            }
        }
        return DP[s1.length][s2.length];
    }


    /**
     * @param sentence (String)
     * @return Returns Tokenized words of the sentence.
     */
    private String[] getWords(String sentence) {
        String[] output = sentence.split("\\s+");
        for (int i=0; i<output.length; i++) {
            output[i] = output[i].replaceAll("[^\\w]", "");
        }
        return output;
    }

    public static void main(String[] args) throws IOException{
        BufferedReader Reader = new BufferedReader(new InputStreamReader(System.in)); 
        String A = Reader.readLine();
        String B = Reader.readLine();

        EditDistance ed = new EditDistance();

        String[] Arr = ed.getWords(A);
        String[] Brr = ed.getWords(B);
        
        boolean ShowChanges = true;

        // Enter if Condition only if the words are  given as an input.
        // Enter else Condition only if Sentences are given as an input.
        if (Arr.length == 1 && Brr.length == 1) {
            System.out.println("Min Edit Distance: "+ ed.getEditCost(Arr[0], Brr[0], ShowChanges));
        }
        else {
            System.out.println("Min Edit Distance: " + ed.getEditCost(Arr, Brr, ShowChanges));
        }
    }
}
