# Cleanup file for making selected SNIa FITRES files pandas-compatible

def fitresCleanup(fname):

    with open(fname, 'r') as f:
        dat = f.readlines()

    varlabels = dat[6]
    dat[12] = varlabels # add variable labels right on top of data

    fname = fname[:-7] + "_selected" + '.FITRES' 
    
    with open(fname, 'w') as newf:       # create new file
        newf.write("%s" % dat[12])
        
        for line in dat[13:]:
            if 'SN' in line:             # remove blank lines
                newf.write("%s" % line)  # write output


# for cleaning up bulk SN file
def simgenFileCleanup(fname):
    with open(fname, 'r') as f:
        dat = f.readlines()

    varlabels = dat[1]
    dat[9] = varlabels               # add variable labels right on top of data

    
    fname = fname[:-4] + "_bulk" + '.DAT' # create processed bulk file
    with open(fname, 'w') as newf:

        newf.write("%s" % dat[9])    # write in the varname header 

        for line in dat[10:]:        # keep only data (no blank lines or sim headers)
            if 'SN' in line:
                newf.write("%s" % line)  # write data output