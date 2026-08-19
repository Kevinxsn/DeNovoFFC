## This script is mainly written by Joel Averbukh @ University of Oxford


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from altair import sequence
from scipy.spatial import cKDTree
import re
import peptide

def extract_parentheses_suffix(s: str):
    """
    If the string ends with (...) — parentheses enclosing any content — 
    return the content inside. Otherwise, return False.
    """
    match = re.search(r"\(([^()]*)\)$", s)
    if match:
        return match.group(1)
    else:
        return False


peptide_table = {
#This is a table relating all peptide prefix names to their corresponding sequences. This is necessary simply because
# the names of input files include these prefixes and not the sequences themselves.
        "ME1": "GEYFGEK",
        "ME2": "(nitro)YEFGIFNQK",
        "ME3": "VVPG(nitro)YGHAVLR",
        "ME4": "LGE(nitro)YGFQNAILVR",
        "ME5": "GR(Me)GRPR",
        "ME6": "GR(Me2)GRPR",
        "ME7": "GR(Me2)GRPR",
        "ME8": "GGNFSGR(Me)GGFGGSR",
        "ME9": "GWGR(Me2)EENLFSWK",
        "ME10": "GMR(Me2)GR(Me2)GR",
        "ME11": "SLGMIFEK(Ac)R",
        "ME12": "SLGMIFEK(Me)3R",
        "ME13": "SLGMIFEKR",
        "ME14": "VTIMPK(Ac)DIQLAR",
        "ME15": "VTIMPK(Me)3DIQLAR",
        "ME16": "VTIMPKDIQLAR",
        "ME17": "TWR(Me2)GGEEK",
        "ME18": "TWR(Me2)GGEEK",
        # "MR1": "Myr-GQELSQHER",
        # "MR2": "Myr-GQDQTK",
        "PH1": "GDFEEIPEEpYLQ",
        "PH2": "EQFEEpYGHMRF-NH2",
        "PH3": "NpYpYGWMDF-NH2",
        "PH4": "EQFDDpYGHMRF-NH2",
        "PH5": "pYGGFL",
        "PH6": "RDpYTGWNleDF-NH2",
        "PH7": "DpYMGWMDF-NH2",
        "PH8": "TWpTLCGTVEY",
        "PH9": "FRGpSGDTSNF",
        "PH10": "FSIAPFpYLDPSNR",
        "PH11": "CLNRQLpSSGVSEIR",
        "PH12": "DADEpYL-NH2",
        # "PH13": "Pyr-DDpSDEEN",
        "PH14": "SApSPEALAFVR",
        "PH15": "SApTPEALAFVR",
        "SU1": "LAIFSC(SO3H)FR",
        "SU2": "C(SO3H)LAGLVF",
        "SU3": "Y(SO3H)GGFL",
        "SU4": "RDY(SO3H)TGW-Nle-DF(NH2)",
        "SU5": "EQFDDY(SO3H)GHMRF(NH2)",
        "SU6": "NY(SO3H)Y(SO3H)GWMDF(NH2)",
        "SU7": "DY(SO3H)MGWMDF(NH2)",
        "SU8": "GDFEEIPEEY(SO3H)LQ",
        "UN1": "GDFEEIPEEYLQ",
        "UN2": "LAIFSCFR",
        "UN3": "TWRGGEEK",
        "UN4": "ESVKEFLA",
        "UN5": "GQELSQHER",
        "UN6": "YGGFL",
        "UN7": "SDGRG",
        "UN8": "KRTLR",
        "UN9": "YGGFLRRIRPKLK",
        "UN10": "PLYKKIIKKLLES",
        "UN14": "GSNKGAIIGLM",
        "UN15": "MLGIIAGKNSG",
        "7255": "ALDLLDRNYLQSLPSK",
        "7732": "KFIFRTAGTAGR",
        "6727": "DQARVAPSSSDPKSKFF",
        "7302": "HGMTVVIRKKF",
        "3510": "GSHQISLDNPDYQQDFFPK"}

define = {
# This is a table relating an amino acid to its monoisotopic mass. It also includes the addition/loss in monoisotopic
# as a result of certain modifications. I have made particular keys for these modifications numbers but really its arbitrary.
# The meanings of these keys can be found and changed in the adjust_peptide() function
          'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
          'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
          'H': 137.08406, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
          'M': 131.04049, 'F': 147.06841, 'P': 97.05276, 'S': 87.03203,
          'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
          '1': 128.09496 + 42.0106 , '2': 156.10111 + 28.0314, '3': 163.06333 + 79.9663,
          '4': 147.06841 - 0.98402, '5': 156.10111 + 28.0314 / 2, '6': 208.0484, '7': 170.1419,
          '8': -0.984016, '9': 101.04768 + 79.9663, ';': 87.03203 + 79.9663, ':': 150.99394,
          '.': 227.02523}


adding = {'NH2': 16.01872, 'H2O': 18.01056}

#The following are dictionaries of neutral loss for various amino acids or modifications. The dictionaries consist of
#the loss as the key, and its monoisotopic mass as the value. The main reason for this interpretation program is to be
#able to create a new neutral loss simply by adding it to an existing dictionary or by creating a new dictionary. All
#these dictionaries include no loss at all -> ''. This is just how I have structured the code.

general_neutral_losses = {'-(H2O)': 18.0106, '-(NH3)': 17.0265, '-(H20)-(NH3)': 18.0106+17.0265, '-2(H2O)': 2*18.0106, '-2(NH3)': 2*17.0265,'': 0}
M_neutral_losses = {'-(CH3CH2SCH3)': 76.0346, '-(CH3SH)': 48.0034, '-(CH2S)': 45.9877, '': 0}
R_neutral_losses = {'-(HN=C=NH)': 42.0218, '': 0}
D_neutral_losses = {'-(HCOOH)': 46.0055, '-(HCOH)': 30.0106, '': 0}
E_neutral_losses = {'-(HCOOH)': 46.0055, '-(HCOH)': 30.0106, '': 0}
pS_neutral_losses = {'-(H3PO4)': 97.9769, '': 0}
pT_neutral_losses = {'-(H3PO4)': 97.9769, '': 0}
pY_neutral_losses = {'-(HPO3)': 79.9663, '': 0}
kAc_neutral_losses = {'-(CH3COOH)': 60.0211, '': 0}
RMe_neutral_losses = {'-(CH3NH2)': 31.0422, '': 0}
RMe2_neutral_losses = {'-(CH3NHCH3)': 45.0578,'-(CH3NH2)': 31.0422, '': 0}
S_neutral_losses = {'-(HCOH)': 30.0106, '': 0}
T_neutral_losses = {'-(HCOH)': 30.0106, '': 0}
Q_neutral_losses = {'-(HCOH)': 30.0106, '': 0}
F_neutral_losses = {'-(HCOH)': 30.0106, '': 0}
CSO3H_neutral_losses = {'-(SO3)': 79.9568, '-(SO2)': 63.9619 ,'': 0}
YSO3H_neutral_losses = {'-(SO3)': 79.9568, '-(SO2)': 63.9619 ,'': 0}

#All the types of losses are then referred to in this losses_dictionary. I have given 'weird' keys for modifications.
#The reason for this is that my code is structures to a single character(residue/mod) to a corresponding neutral loss
#dictionary and so I wanted to keep all keys as single characters

losses_dictionary = {'M': M_neutral_losses, 'R': R_neutral_losses, 'D': D_neutral_losses, 'E': E_neutral_losses,
                     ';': pS_neutral_losses, '9': pT_neutral_losses, '3': pY_neutral_losses, '1': kAc_neutral_losses,
                     '5': RMe_neutral_losses, '2': RMe2_neutral_losses, 'S': S_neutral_losses, 'T': T_neutral_losses,
                     'Q': Q_neutral_losses, 'F': F_neutral_losses, ':': CSO3H_neutral_losses, '.': YSO3H_neutral_losses}

def adjust_peptide(peptide_input):
    #This function finds modifications a peptide sequence and replaces them with the character that corresponds to its
    #neutral losses dictionary as found in losses_dictionary as well as its mono mass in define.
    peptide_input = peptide_input.replace(' ','')
    peptide_input = peptide_input.replace('K(Ac)', '1')
    peptide_input = peptide_input.replace('R(Me2)', '2')
    peptide_input = peptide_input.replace('Y(p)', '3')
    #peptide_input = peptide_input.replace('F(NH2)', '4')

    # 21/12/2022
    peptide_input = peptide_input.replace('R(Me)', '5')

    # 19/09/2025
    #peptide_input = peptide_input.replace('(nitro)Y', '6')
    peptide_input = peptide_input.replace('Y(nitro)', '6')
    #peptide_input = peptide_input.replace('K(Me)3', '7')
    peptide_input = peptide_input.replace('pY', '3')
    #peptide_input = peptide_input.replace('-NH2', '8')
    peptide_input = peptide_input.replace('Nle', 'L')
    peptide_input = peptide_input.replace('-Nle-', 'L')
    peptide_input = peptide_input.replace('pT', '9')
    peptide_input = peptide_input.replace('T(p)', '9')
    peptide_input = peptide_input.replace('pS', ';')
    peptide_input = peptide_input.replace('C(SO3H)', ':')
    peptide_input = peptide_input.replace('Y(SO3H)', '.')
    #peptide_input = peptide_input.replace('(NH2)', '8')
    ending = extract_parentheses_suffix(peptide_input)
    if ending:
        peptide_input = peptide_input.replace('(' + ending + ')', '')
    
    
    return peptide_input, ending

def combine_dicts(d1, d2):
    #This function takes two neutral loss dictionaries and produces one single dictionary with all possible neutral loss
    # combinations
    combined = {}
    for k1, v1 in d1.items():
        for k2, v2 in d2.items():
            if (k1.count('-') == 1 and k2.count('-') == 1) or (k1.count('-') == 1 and k2 == '') or (k2.count('-') == 1 and k1 == '') or (k1 == '' and k2 == ''): #prevents more than two neutral losses - and still allows only one neutral loss or zero
                new_key = k1 + k2     # concatenates losses
                new_val = v1 + v2     # add mono mass of losses
                combined[new_key] = new_val
    return combined

def create_combined_dict_from_sequence(sequence):
    #This function finds all residues/modifications in losses_dicionary that are present in a sequence
    #and creates a combined neutral losses dictionary
    combined_dict = general_neutral_losses
    for residue, dict in losses_dictionary.items():
        if residue in sequence:
            combined_dict = combine_dicts(combined_dict, dict)

    return combined_dict

def create_all_possible_neutral_loss_combinations(ion_1, ion_2, charge_1, charge_2, neutral_losses_1, neutral_losses_2, mono_mass_1, mono_mass_2):
    #This function creates all possible pairs of combinations between two ions (b,y etc.) plus possible neutral losses
    #The M/Zs for each pair are calculated using the mono masses of the ions and their possible neutral losses
    pairs = []
    for loss1, mass1 in neutral_losses_1.items():
        for loss2, mass2 in neutral_losses_2.items():
            pairs.append([ion_1 + loss1 + '(' + str(charge_1) + '+)',
                          (mono_mass_1 - mass1 + (charge_1) * 1.00784)/charge_1,
                          (mono_mass_2 - mass2 + (charge_2) * 1.00784)/charge_2,
                          ion_2 + loss2 + '(' + str(charge_2) + '+)'])

    return pairs

def all_pairs(sequence, ending, charge, sequence_mono_array):
    #This is the main function which produces all possible types of ion pairs including neutral losses that may be possible
    #due to a residue or modification being present in an ion. In this code, I have only included b ions, b internal ions
    # a ions, a internal ions and y ions, however adding new possible pairs amounts to copying and pasting existing code
    # and simply adjusting masses and names
    
    if ending == False:
        ending = adding['H2O']
    else:
        ending = adding[ending]
    
    all_pairs = []
    for i in range(0,len(sequence)):
        for j in range(0,len(sequence)-i-1): # range of j makes sure there can be no overlaps - i.e no b,y ions cross over
            for c1 in range(1, charge):
                for c2 in range(1, charge):
                    if (c1 <= i+1 and c2 <= j+1) and ((i + j + 2 == len(sequence) and c1 + c2 == charge) or (i + j + 2 < len(sequence) and c1 + c2 <= charge)):
                        # complementary and uncomplementary pairs where charge has to be smaller than or equal to sequence length
                        b_ion = 'b' + str(i+1)
                        b_seq = sequence[:i+1]
                        b_mass = sum(sequence_mono_array[:i+1])
                        b_dict = create_combined_dict_from_sequence(b_seq)

                        a_ion = 'a' + str(i+1)
                        a_seq = sequence[:i+1]
                        a_mass = b_mass - 28
                        a_dict = create_combined_dict_from_sequence(a_seq)

                        y_ion = 'y' + str(j+1)
                        y_seq = sequence[-j-1:]
                        #y_mass = sum(sequence_mono_array[-j-1:]) + 18.01056
                        y_mass = sum(sequence_mono_array[-j-1:]) + ending
                        y_dict = create_combined_dict_from_sequence(y_seq)

                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(b_ion, y_ion,
                        c1, c2, b_dict, y_dict, b_mass, y_mass)  # b-y
                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(a_ion, y_ion,
                        c1, c2, a_dict, y_dict, a_mass, y_mass)  # a-y

                        b_internal_ion = 'bi(' + str(i+2) + '-' + str(len(sequence) - j - 1) + ')'
                        b_internal_seq = sequence[i+1:len(sequence) - j-1]
                        b_internal_mass = sum(sequence_mono_array[i+1:len(sequence) - j-1])
                        b_internal_dict = create_combined_dict_from_sequence(b_internal_seq)

                        a_internal_ion = 'ai(' + str(i+2) + '-' + str(len(sequence) - j - 1) + ')'
                        a_internal_seq = sequence[i+1:len(sequence) - j-1]
                        a_internal_mass = sum(sequence_mono_array[i+1:len(sequence) - j-1]) - 28
                        a_internal_dict = create_combined_dict_from_sequence(a_internal_seq)

                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(b_internal_ion, y_ion,
                        c1, c2, b_internal_dict, y_dict, b_internal_mass, y_mass)  # bi-y
                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(a_internal_ion, y_ion,
                        c1, c2, a_internal_dict, y_dict, a_internal_mass, y_mass)  # ai-y

                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(b_ion, b_internal_ion,
                        c1, c2, b_dict, b_internal_dict, b_mass, b_internal_mass)  # b-bi
                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(b_ion, a_internal_ion,
                        c1, c2, b_dict, a_internal_dict, b_mass, a_internal_mass)  # b-ai
                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(a_ion, b_internal_ion,
                        c1, c2, a_dict, b_internal_dict, a_mass, b_internal_mass)  # a-bi
                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(a_ion, a_internal_ion,
                        c1, c2, a_dict, a_internal_dict, a_mass, a_internal_mass)  # a-ai

                    elif c1 + c2 <= charge and c1 <= len(sequence) - (i+1) - (j+1) and c2 <= j + 1:
                        #if b_internal_ion and y_ion have valid charges
                        y_ion = 'y' + str(j + 1)
                        y_seq = sequence[-j - 1:]
                        y_mass = sum(sequence_mono_array[-j - 1:]) + 18.01056
                        y_dict = create_combined_dict_from_sequence(y_seq)

                        b_internal_ion = 'bi(' + str(i + 2) + '-' + str(len(sequence) - j - 1) + ')'
                        b_internal_seq = sequence[i + 1:len(sequence) - j - 1]
                        b_internal_mass = sum(sequence_mono_array[i + 1:len(sequence) - j - 1])
                        b_internal_dict = create_combined_dict_from_sequence(b_internal_seq)

                        a_internal_ion = 'ai(' + str(i + 2) + '-' + str(len(sequence) - j - 1) + ')'
                        a_internal_seq = sequence[i + 1:len(sequence) - j - 1]
                        a_internal_mass = sum(sequence_mono_array[i + 1:len(sequence) - j - 1]) - 28
                        a_internal_dict = create_combined_dict_from_sequence(a_internal_seq)

                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(b_internal_ion, y_ion,
                        c1, c2, b_internal_dict, y_dict, b_internal_mass, y_mass)  # bi-y
                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(a_internal_ion, y_ion,
                        c1, c2, a_internal_dict, y_dict, a_internal_mass, y_mass)  # ai-y

                    elif c1 + c2 <= charge and c1 <= len(sequence) - (i+1) - (j+1) and c1 <= i + 1:
                        #if b_internal_ion and b_ion have valid charges
                        b_internal_ion = 'bi(' + str(i + 2) + '-' + str(len(sequence) - j - 1) + ')'
                        b_internal_seq = sequence[i + 1:len(sequence) - j - 1]
                        b_internal_mass = sum(sequence_mono_array[i + 1:len(sequence) - j - 1])
                        b_internal_dict = create_combined_dict_from_sequence(b_internal_seq)

                        a_internal_ion = 'ai(' + str(i + 2) + '-' + str(len(sequence) - j - 1) + ')'
                        a_internal_seq = sequence[i + 1:len(sequence) - j - 1]
                        a_internal_mass = sum(sequence_mono_array[i + 1:len(sequence) - j - 1]) - 28
                        a_internal_dict = create_combined_dict_from_sequence(a_internal_seq)

                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(b_ion, b_internal_ion,
                        c1, c2, b_dict, b_internal_dict, b_mass, b_internal_mass)  # b-bi
                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(b_ion, a_internal_ion,
                        c1, c2, b_dict, a_internal_dict, b_mass, a_internal_mass)  # b-ai
                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(a_ion, b_internal_ion,
                        c1, c2, a_dict, b_internal_dict, a_mass, b_internal_mass)  # a-bi
                        all_pairs = all_pairs + create_all_possible_neutral_loss_combinations(a_ion, a_internal_ion,
                        c1, c2, a_dict, a_internal_dict, a_mass, a_internal_mass)  # a-ai

    return all_pairs


def extract_prefix_and_charge(s):
    #This function simply extracts the sequence prefix and its charge from a specific type of filename system -
    # namely '20160629_1557_ME15_2+_CVscan_NCE25' which is for LTQXL files
    # so for different input files this will become obsolete

    # Find all positions of underscores
    underscores = [i for i, char in enumerate(s) if char == '_']

    if len(underscores) < 3:
        return "", ""  # Not enough underscores

    # Substring between 2nd and 3rd underscore
    between = s[underscores[1] + 1: underscores[2]]

    # First letter after 3rd underscore
    after = s[underscores[2] + 1] if underscores[2] + 1 < len(s) else ""

    return between, after

def report(prefix, sequence_pairs):

    #This function is the actual interpreter. Sequence pairs are all the possible pairs of ions along with their M/Zs
    #and now this function tries to match these M/Zs with ones in a top features .npy array file of M/Z pairs. Naturally,
    #there is a tolerance for a valid pair and multiple interpretations are allowed for.

    '''data = extract_prefix_and_charge(prefix)

    peptide_input = peptide_table[data[0]]  # peptide_table[prefix[0:4] ]   #input("Enter the theoretical sequence of peptide:\n ")
    charge_input = int(data[1])  # 3 #input("Enter the charge of peptide:\n") # input '3'''''

    #The following is just the file name path to the _topfeat.npy array which holds the M/Z uninterpreted topfeatures

    path = 'G:/2DPCMS/Linear ion trap - Intensity Normalisation - No PCov/'  #
    name = prefix  # '7732_2D-PC-MS_0-1pmol-ul_0-1AGC_0-7quadiso' #CHANGED

    #top_feat = np.load(path + name + '/tic_topfeat.npy')

    top_feat = np.load(path + name + '/_topfeat.npy')

    tolerance = 0.8  # for linear ion trap

    '''peptide_input = adjust_peptide(peptide_input)

    peptide_list = list(peptide_input)  # list of amino acids in peptide
    peptide = []  # creates list of mono masses for each amino acid in peptide
    for item in peptide_list:
        peptide.append(define[item])'''


    #sequence_pairs = all_pairs(peptide_input, charge_input, peptide)

    # create a pandas dataframe to add title for each column and round the m/z values
    df_ref = pd.DataFrame(sequence_pairs, columns=['Interpretation A', 'm/z A', 'm/z B', 'Interpretation B']) #, 'Plausibility'])
    df_ref['m/z A'] = df_ref['m/z A'].round(2)
    df_ref['m/z B'] = df_ref['m/z B'].round(2)
    sequence_pairs = df_ref.to_numpy()

    correlation = []
    ## cut off diagonal with 5.5 Da - removes all pairs found within 5.5 Da
    new_top_feat = []
    for i in top_feat:
        if abs(i[0] - i[1]) >= 7: # was 5.5
            new_top_feat.append(i)
    new_top_feat = np.array(new_top_feat)  # so new_top_feat removes all of the pairings within 5.5 Da!
    # display(top_feat)

    # Coordinates of top features
    top_coords = np.array([[row[0], row[1]] for row in new_top_feat])

    #The following code tries to find matching M/Z pairs in sequence_pairs that are in new_top_feat by by testing if
    # M/Z(1):M/Z(2) or M/Z(2):M/Z(1) exists in sequence pairs (since order is arbitrary). The tolerance is used. This
    # uses a cKDTree which searches for M/Z pairs by organising them into a binary tree - faster than looking through all!

    # Orientation 1: i[0] ↔ j[2], i[1] ↔ j[1]
    seq_coords1 = np.array([[float(row[2]), float(row[1])] for row in sequence_pairs])
    tree1 = cKDTree(seq_coords1)
    matches_indices1 = tree1.query_ball_point(top_coords, r=tolerance)

    for i_idx, j_idxs in enumerate(matches_indices1):
        i_row = new_top_feat[i_idx]
        for j_idx in j_idxs:
            j_row = sequence_pairs[j_idx]
            mass_deviation = np.sqrt((i_row[0] - float(j_row[2])) ** 2 + (i_row[1] - float(j_row[1])) ** 2)
            correlation.append([
                str(j_row[3]),
                i_row[0],
                i_row[1],
                str(j_row[0]),
                i_row[2],
                i_row[3],
                round(mass_deviation, 3)
            ])

    # Orientation 2: i[0] ↔ j[1], i[1] ↔ j[2]
    seq_coords2 = np.array([[float(row[1]), float(row[2])] for row in sequence_pairs])
    tree2 = cKDTree(seq_coords2)
    matches_indices2 = tree2.query_ball_point(top_coords, r=tolerance)

    for i_idx, j_idxs in enumerate(matches_indices2):
        i_row = new_top_feat[i_idx]
        for j_idx in j_idxs:
            j_row = sequence_pairs[j_idx]
            mass_deviation = np.sqrt((i_row[0] - float(j_row[1])) ** 2 + (i_row[1] - float(j_row[2])) ** 2)
            correlation.append([
                str(j_row[0]),
                i_row[0],
                i_row[1],
                str(j_row[3]),
                i_row[2],
                i_row[3],
                round(mass_deviation, 3)
            ])

    # compare the measured fragments pairs to theoretical values list
    # if the difference is smaller than or equal to the tolerance,
    # the array created will record this pair
    # tolerance is 0.8
    '''for i in new_top_feat:
        for j in sequence_pairs:
            if abs(i[0] - j[2]) <= tolerance and abs(i[1] - j[1]) <= tolerance:
                mass_deviation = np.sqrt(abs(i[0] - j[2]) ** 2 + abs(i[1] - j[1]) ** 2)
                correlation.append([str(j[3]), i[0], i[1], str(j[0]), i[2], i[3], mass_deviation.round(3)])
            elif abs(i[0] - j[1]) <= tolerance and abs(i[1] - j[2]) <= tolerance:
                mass_deviation = np.sqrt(abs(i[0] - j[1]) ** 2 + abs(i[1] - j[2]) ** 2)
                correlation.append([str(j[0]), i[0], i[1], str(j[3]), i[2], i[3], mass_deviation.round(3)])'''

    correlation = np.array(correlation)
    correlation[:, 1:3]  # all rows - take only column 1 and column 2
    correlation_list = correlation[:, 1:3].astype(float).tolist()

    # if there is no match with the theoretical values, will record the pair as unidentified fragments
    unidentify = []
    for item in new_top_feat.tolist():
        if item[:2] not in correlation_list:
            unidentify.append(['', item[0], item[1], '', item[2], item[3], ''])  #, ''])

    correlation_new = correlation.tolist() + unidentify
    correlation_new = np.array(correlation_new)

    # sort the fragments according to their normalised correlation scores
    mapp = np.array([float(x) for x in correlation_new[:, 5]])
    correlation_new = correlation_new[np.flip(mapp.argsort())]

    df = pd.DataFrame(correlation_new, columns=['Interpretation A', 'm/z A', 'm/z B', 'Interpretation B',
                                                'CorrelationScore', 'NormalisedScore', 'MassDeviation'])#'Plausibility', 'MassDeviation'])

    df.NormalisedScore = df.NormalisedScore.astype(float)

    # sort the fragments according to normalisation scores and then to their interpretation plausibility and then to
    # their mass deviation from theoretical values
    #df2 = df.sort_values(by=['NormalisedScore', 'Plausibility', 'MassDeviation'], ascending=[False, True, True])
    df2 = df.sort_values(by=['NormalisedScore', 'MassDeviation'], ascending=[False, True])

    correlation_new = df2.to_numpy()

    # add index number to fragments
    # different index numbers refer to different pairs
    plus_index = []
    number = 1
    for row in range(len(correlation_new)):
        if row < 1:
            plus_index.append(np.insert(correlation_new[row], 0, number))
        elif row >= 1:
            if correlation_new[row][1] == correlation_new[row - 1][1] and correlation_new[row][2] == \
                    correlation_new[row - 1][2]:
                plus_index.append(np.insert(correlation_new[row], 0, ''))

            else:
                number = number + 1
                plus_index.append(np.insert(correlation_new[row], 0, number))

    plus_index.insert(0, ['Index', 'Interpretation A', 'm/z A', 'm/z B', 'Interpretation B',
                          'CorrelationScore', 'NormalisedScore', 'MassDeviation']) #'Plausibility', 'MassDeviation'])

    # save the file
    name_of_file = path + str(name) + '/' + str(name) + 'new' + '.csv'
    npy_file = path + str(name) + '/interpretation.npy'
    np.savetxt(name_of_file, plus_index, delimiter=',', fmt='%s')
    np.save(npy_file, plus_index)




def report_from_dataframe(df_pairs,
                          sequence_pairs,
                          tolerance=0.8,
                          diagonal_cutoff=7.0,
                          round_theory_to=2):
    """
    Match experimental m/z pairs (df_pairs with columns ['m/z A','m/z B'])
    against theoretical 'sequence_pairs' produced by all_pairs(...),
    and return an annotated DataFrame like the original report() produced.

    Optional: you can pass columns 'CorrelationScore' and 'NormalisedScore'
    in df_pairs; otherwise they default to 1.0.
    """

    # --- 1) Normalize experimental input ---
    df_exp = df_pairs.copy()
    # Ensure required columns exist
    if 'm/z A' not in df_exp.columns or 'm/z B' not in df_exp.columns:
        # If your columns are unnamed (e.g., 0 and 1), rename them:
        # df_exp = df_exp.rename(columns={df_exp.columns[0]: 'm/z A',
        #                                 df_exp.columns[1]: 'm/z B'})
        raise ValueError("df_pairs must have columns 'm/z A' and 'm/z B'.")

    # If correlation/normalised scores missing, set defaults
    if 'CorrelationScore' not in df_exp.columns:
        df_exp['CorrelationScore'] = 1.0
    if 'NormalisedScore' not in df_exp.columns:
        df_exp['NormalisedScore'] = 1.0

    # Remove near-diagonal pairs (instrument heuristic)
    df_exp = df_exp.loc[(df_exp['m/z A'] - df_exp['m/z B']).abs() >= diagonal_cutoff].copy()

    # --- 2) Prepare theoretical pairs ---
    df_ref = pd.DataFrame(sequence_pairs,
                          columns=['Interpretation A', 'm/z A', 'm/z B', 'Interpretation B']).copy()
    if round_theory_to is not None:
        df_ref['m/z A'] = df_ref['m/z A'].round(round_theory_to)
        df_ref['m/z B'] = df_ref['m/z B'].round(round_theory_to)

    # Build KD-trees on theoretical coordinates for both orientations
    seq_coords1 = df_ref[['m/z B', 'm/z A']].to_numpy(dtype=float)  # (theoryB, theoryA)
    seq_coords2 = df_ref[['m/z A', 'm/z B']].to_numpy(dtype=float)  # (theoryA, theoryB)
    tree1 = cKDTree(seq_coords1)
    tree2 = cKDTree(seq_coords2)

    # Experimental coords
    top_coords = df_exp[['m/z A', 'm/z B']].to_numpy(dtype=float)

    # --- 3) KD-tree radius matches ---
    correlation = []

    # Orientation 1: exp(A,B) ~ (theoryB, theoryA)
    matches_indices1 = tree1.query_ball_point(top_coords, r=tolerance)
    for i_idx, j_idxs in enumerate(matches_indices1):
        exp_row = df_exp.iloc[i_idx]
        for j_idx in j_idxs:
            theo_row = df_ref.iloc[j_idx]
            # Euclidean mass deviation in the 2D plane
            md = np.hypot(exp_row['m/z A'] - float(theo_row['m/z B']),
                          exp_row['m/z B'] - float(theo_row['m/z A']))
            correlation.append([
                theo_row['Interpretation B'],                    # Interpretation A (left) in this orientation
                exp_row['m/z A'], exp_row['m/z B'],
                theo_row['Interpretation A'],                    # Interpretation B (right)
                exp_row['CorrelationScore'], exp_row['NormalisedScore'],
                round(float(md), 3)
            ])

    # Orientation 2: exp(A,B) ~ (theoryA, theoryB)
    matches_indices2 = tree2.query_ball_point(top_coords, r=tolerance)
    for i_idx, j_idxs in enumerate(matches_indices2):
        exp_row = df_exp.iloc[i_idx]
        for j_idx in j_idxs:
            theo_row = df_ref.iloc[j_idx]
            md = np.hypot(exp_row['m/z A'] - float(theo_row['m/z A']),
                          exp_row['m/z B'] - float(theo_row['m/z B']))
            correlation.append([
                theo_row['Interpretation A'],
                exp_row['m/z A'], exp_row['m/z B'],
                theo_row['Interpretation B'],
                exp_row['CorrelationScore'], exp_row['NormalisedScore'],
                round(float(md), 3)
            ])

    # --- 4) Add "unidentified" experimental pairs ---
    corr_np = np.array(correlation, dtype=object) if correlation else np.empty((0,7), dtype=object)
    matched_coords = set()
    for row in corr_np.tolist():
        matched_coords.add((float(row[1]), float(row[2])))

    unidentify = []
    for _, row in df_exp.iterrows():
        key = (float(row['m/z A']), float(row['m/z B']))
        if key not in matched_coords:
            unidentify.append(['', row['m/z A'], row['m/z B'], '',
                               row['CorrelationScore'], row['NormalisedScore'], ''])

    correlation_new = (corr_np.tolist() if correlation else []) + unidentify
    df_out = pd.DataFrame(correlation_new,
                          columns=['Interpretation A', 'm/z A', 'm/z B', 'Interpretation B',
                                   'CorrelationScore', 'NormalisedScore', 'MassDeviation'])

    # Types
    df_out['NormalisedScore'] = pd.to_numeric(df_out['NormalisedScore'], errors='coerce').fillna(0.0)
    # Sort best first: high NormalisedScore, then small MassDeviation (treat '' as inf)
    md_numeric = pd.to_numeric(df_out['MassDeviation'], errors='coerce')
    df_out = df_out.assign(_md=md_numeric.fillna(np.inf)) \
                   .sort_values(by=['NormalisedScore', '_md'], ascending=[False, True]) \
                   .drop(columns=['_md'])

    # --- 5) Add group index per unique experimental (A,B) ---
    out_np = df_out.to_numpy(dtype=object)
    indexed = []
    idx = 0
    prev = (None, None)
    for r in out_np:
        cur = (r[1], r[2])
        if cur != prev:
            idx += 1
            indexed.append([idx] + list(r))
            prev = cur
        else:
            indexed.append([''] + list(r))

    df_indexed = pd.DataFrame(indexed, columns=[
        'Index', 'Interpretation A', 'm/z A', 'm/z B', 'Interpretation B',
        'CorrelationScore', 'NormalisedScore', 'MassDeviation'
    ])

    return df_indexed


def report_data(df_pairs, input_sequence, charge, tolerance_input=0.8, merge_original = False):
    
    
    ## df_pairs: dataframe with m/z A, m/z B column
    ## input_sequence: 'GGNFSGR(Me)GGFGGSR'
    ## charge: charge of the peptide
    
    peptide_input = adjust_peptide(input_sequence)
    peptide_list = list(peptide_input[0])
    peptide = []
    for item in peptide_list:
        peptide.append(define[item])
    sequence_pairs = all_pairs(peptide_input[0], peptide_input[1], charge, peptide)
    df_result = report_from_dataframe(df_pairs, sequence_pairs, tolerance=tolerance_input)
    
    if merge_original:
        merged = pd.merge(df_pairs, df_result, on=['m/z A', 'm/z B'], how='outer')
        return merged[['each_original_data', 'm/z A', 'm/z B', 'Interpretation A', 'Interpretation B', 'CorrelationScore', 'NormalisedScore', 'MassDeviation']]
    
    return df_result

def report_data_with_ranking(df_pairs, input_sequence, charge, tolerance_input=0.8, merge_original = False):
    
    
    ## df_pairs: dataframe with m/z A, m/z B column
    ## input_sequence: 'GGNFSGR(Me)GGFGGSR'
    ## charge: charge of the peptide
    
    peptide_input = adjust_peptide(input_sequence)
    peptide_list = list(peptide_input[0])
    peptide = []
    for item in peptide_list:
        peptide.append(define[item])
    sequence_pairs = all_pairs(peptide_input[0], peptide_input[1], charge, peptide)
    df_result = report_from_dataframe(df_pairs, sequence_pairs, tolerance=tolerance_input)
    
    if merge_original:
        merged = pd.merge(df_pairs, df_result, on=['m/z A', 'm/z B'], how='outer')
        return merged[['each_original_data', 'm/z A', 'm/z B', 'Interpretation A', 'Interpretation B', 'CorrelationScore', 'NormalisedScore', 'MassDeviation', 'Ranking']]
    
    return df_result
    


if __name__ == "__main__":
    pep_seq = 'LGEY(nitro)GFQNAILVR'
    data = 6
    df = pd.read_csv(f'data/data_table/data_sheet{data}.csv')
    charge = 3
    
    df_mass = df[['each_original_data', 'mass1', 'mass2']]
    df_mass_clean = df_mass.dropna()
    df_mass_clean.columns = ['each_original_data', 'm/z A', 'm/z B']
    df_result = report_data(df_mass_clean, pep_seq, charge, tolerance_input=1, merge_original=True)
    print(df_result.head(30))
    df_result.to_csv(f'data/annotation/data{data}.csv')

