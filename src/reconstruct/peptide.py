
## this script will transfer a raw sequence into the array of AA




import re
from typing import List, Optional

class AA:
    # Class-level dictionaries (shared by all instances)
    amino_acid_masses = {
        "A": 71.03711,   # Alanine
        "R": 156.10111,  # Arginine
        "N": 114.04293,  # Asparagine
        "D": 115.02694,  # Aspartic acid
        "C": 103.00919,  # Cysteine
        "E": 129.04259,  # Glutamic acid
        "Q": 128.05858,  # Glutamine
        "G": 57.02146,   # Glycine
        "H": 137.05891,  # Histidine
        "I": 113.08406,  # Isoleucine
        "L": 113.08406,  # Leucine
        "K": 128.09496,  # Lysine
        "M": 131.04049,  # Methionine
        "F": 147.06841,  # Phenylalanine
        "P": 97.05276,   # Proline
        "S": 87.03203,   # Serine
        "T": 101.04768,  # Threonine
        "W": 186.07931,  # Tryptophan
        "Y": 163.06333,  # Tyrosine
        "V": 99.06841    # Valine
    }
    
    ptm_masses = {
        "Me": 14.01565,          # Methylation (+CH3)
        "Me2": 28.03130,         # Dimethylation (+2×CH3)
        "Me3": 42.04695,         # Trimethylation (+3×CH3)
        "p": 79.96633,           # Phosphorylation (+HPO3)
        "nitro": 44.98508,       # Nitration (+NO2)
        "NH2": 16.01872,
        #"P": 79.96633,

        "Ac": 42.01056,          # Acetylation (+COCH3)
        "Ox": 15.99491,          # Oxidation (commonly Met→MetO)
        "Carbamidomethyl": 57.02146, # IAA alkylation on cysteine
        "Carboxymethyl": 58.00548,   # Carboxymethylation on cysteine

        "GlyGly": 114.04293,     # Diglycine (ubiquitin remnant)
        "Ubi": 114.04293,        # Same as GlyGly, alternative notation

        "Formyl": 27.99491,      # Formylation (+CHO)
        "Deamidation": 0.98402,  # N/Q → D/E
        "Pyro-Glu (Q)": -17.02655, # N-terminal Gln → pyroGlu
        "Pyro-Glu (E)": -18.01056, # N-terminal Glu → pyroGlu
    }
    
    
    element_masses = {
        'proton':1.00725,
        'H2O':18.01056,
        'carbonyl':28.0106,
        'NH3':17.02655,
        'NH2':17.02655 - 1.00725
    }
    
    def __init__(self, acid_name, attach=None):
        if attach is not None:
            attach = attach.replace('(', '').replace(')', '').replace(' ','')
        self.acid_name = acid_name   # e.g., "K"
        self.attach = attach         # e.g., "Me"
        
    def get_mass(self):
        base_mass = AA.amino_acid_masses.get(self.acid_name, 0)
        ptm_mass = AA.ptm_masses.get(self.attach, 0) if self.attach else 0
        return base_mass + ptm_mass
    
    def __repr__(self):
        return f"{self.acid_name}({self.attach})" if self.attach else self.acid_name
    











class Pep:
    
    def __init__(self, raw_seq, end_h20 = True):
        self.end_h20 = end_h20
        self.seq = Pep.extract_sequence(raw_seq) # the sequence without modification
        self.rev_seq = self.seq[::-1]
        self.seq_mod, self.ion, self.charge = Pep.parse_peptide(raw_seq) ## the sequence with modification
        self.pep_len = len(self.seq)
        self.AA_array = Pep.parse_modified_sequence(self.seq, self.seq_mod, AA.ptm_masses) ## the array store AA
        self.rev_AA_array = self.AA_array[::-1]
        if len(self.AA_array) != self.pep_len:
            print('AA array length and sequence length are unequal, please check')
        #print('the AA array with modification is:', print(self.AA_array))
        self.pep_mass = 0
        for aa in self.AA_array:
            self.pep_mass += aa.get_mass()
        if end_h20 == True:
            self.pep_mass += AA.element_masses['H2O']
        else:
            self.pep_mass = self.pep_mass + AA.element_masses[end_h20]
        h_num = int(self.charge.replace('+','').replace('H',''))
        self.pep_mass += h_num * AA.element_masses['proton']
        self.seq_mass = self.pep_mass - h_num * AA.element_masses['proton'] - AA.element_masses['H2O']
        
        
    def extract_sequence(peptide: str) -> str:
        """
        Remove modifications (e.g. Me), charge info (e.g. +2H, 2+),
        and keep only the plain amino acid sequence.
        
        Example:
        [GGNFSGRMeGGFGGSR+2H]2+  -->  GGNFSGRGGFGGSR
        """
        # 1. Remove surrounding brackets if present
        peptide = peptide.replace('(P)', '')
        peptide = peptide.replace('(nitro)', '')
        peptide = peptide.replace('(Me2)', '')
        peptide = peptide.replace('Me', '')
        peptide = peptide.replace('Ac', '')
        peptide = peptide.replace('(p)', '')
        peptide = peptide.replace('(NH2)', '')
        hydro_pattern = r'\+\d+H'
        peptide = re.sub(hydro_pattern, '', peptide)
        
        peptide = peptide.strip("[]")
        
        # 2. Remove modification tags like "Me", "Ox", "Ac" (assuming uppercase letters only are residues)
        # Keep only capital letters A–Z
        seq = re.sub(r'[^A-Z]', '', peptide)
        
        return seq
    
    
    
    def parse_peptide(peptide_str: str):
        """
        Parse a peptide string of the form:
        [SEQUENCE+ION]CHARGE
        
        Example:
        [GGNFSGRMeGGFGGSR+2H]2+  -> ('GGNFSGRMeGGFGGSR', '+2H', '2+')
        [VTIMPKAcDIQLAR+3H]3+    -> ('VTIMPKAcDIQLAR', '+3H', '3+')
        """
        # 1. Remove whitespace
        peptide_str = "".join(peptide_str.split())
        
        # 2. Regex match
        
        
        
        #pattern = re.compile(r"\[(?P<seq>[A-Za-z]+)(?P<ion>\+\d+H)\](?P<charge>\d+\+)")
        
        #match = pattern.match(peptide_str)
        #if not match:
        #    raise ValueError(f"Invalid peptide format: {peptide_str}")
        
        #return match.group("seq"), match.group("ion"), match.group("charge")
        
        pattern = re.compile(
            r"\[(?P<seq>[A-Za-z0-9()\-\_]+)"  # allow AA letters + inline/() mods (Me, Me2, Ac, nitro, (p), etc.)
            r"(?P<ion>\+\d+H)\]"              # +2H, +3H, ...
            r"(?P<charge>\d+\+)$"             # 2+, 3+, ...
        )

        m = pattern.fullmatch(peptide_str)
        if not m:
            raise ValueError(f"Invalid peptide format: {peptide_str}")

        return m.group("seq"), m.group("ion"), m.group("charge")
            
        
        
        
    def parse_modified_sequence(clean_seq: str, mod_seq: str, ptm_dict: dict) -> List[AA]:
        """
        Build an AA list by aligning a clean sequence with its modified representation.
        - clean_seq: e.g. "GGNFSGRGGFGGSR"
        - mod_seq:   e.g. "GGNFSGRMeGGFGGSR" or "EQFDDY(p)GHMRF(NH2)" or "VTIMPKAcDIQLAR"
        - ptm_dict:  keys are valid PTM names, e.g. {"Me": 14.01565, "Me2": 28.03, "Ac": 42.01, "p": 79.97, "NH2": 16.02, ...}

        Returns: [AA(...), AA(...), ...] aligned to clean_seq
        """

        # 0) Precompute helpers
        mod_seq = "".join(mod_seq.split())  # remove whitespace
        # Sort PTM keys by length desc for greedy longest match (avoids "Me" matching before "Me2")
        ptm_keys_sorted = sorted(ptm_dict.keys(), key=len, reverse=True)

        def try_match_inline_ptm(s: str, start: int):
            """
            Try to match an inline PTM (not in parentheses) at position 'start' using ptm_dict keys.
            Returns (ptm_str, new_pos) if matched, else (None, start).
            """
            for key in ptm_keys_sorted:
                if s.startswith(key, start):
                    return key, start + len(key)
            return None, start

        def try_match_parenthetical_ptm(s: str, start: int):
            """
            If the next token is '(...)', and the inside is a known PTM key, consume it.
            Returns (ptm_str, new_pos) if matched, else (None, start).
            """
            if start < len(s) and s[start] == '(':
                m = re.match(r"\(([^)]+)\)", s[start:])
                if m:
                    inside = m.group(1)
                    if inside in ptm_dict:
                        return inside, start + len(m.group(0))
            return None, start

        # 1) Tokenize the modified sequence into [(AA, [PTMs...]), ...]
        tokens = []
        j = 0
        while j < len(mod_seq):
            ch = mod_seq[j]

            # We expect uppercase amino-acid letters to mark residues (A–Z)
            if 'A' <= ch <= 'Z':
                aa = ch
                j += 1

                attached_ptms = []

                # Consume any number of parenthetical PTMs immediately following, if present
                while True:
                    ptm, j_new = try_match_parenthetical_ptm(mod_seq, j)
                    if ptm is None:
                        break
                    attached_ptms.append(ptm)
                    j = j_new

                # Then consume any number of inline PTMs (greedy by length) immediately following
                while True:
                    ptm, j_new = try_match_inline_ptm(mod_seq, j)
                    if ptm is None:
                        break
                    attached_ptms.append(ptm)
                    j = j_new

                tokens.append((aa, attached_ptms))

            else:
                # Non-uppercase characters outside the patterns are ignored (e.g., brackets from other contexts)
                j += 1

        # 2) Validate alignment with clean_seq
        if len(tokens) != len(clean_seq):
            raise ValueError(
                f"Residue count mismatch: clean_seq has {len(clean_seq)} AAs, the clean_seq is {clean_seq}"
                f"but modified sequence parsed into {len(tokens)} residues, the modified seq is {tokens}"

            )

        for idx, (aa, _ptms) in enumerate(tokens):
            if aa != clean_seq[idx]:
                raise ValueError(
                    f"Residue mismatch at position {idx}: clean '{clean_seq[idx]}' vs modified '{aa}'."
                )

        # 3) Build AA objects; if multiple PTMs attached, join with '+' (or change to a list if you prefer)
        aa_list = []
        for aa, ptms in tokens:
            attach = "+".join(ptms) if ptms else None
            aa_list.append(AA(aa, attach))

        return aa_list
    
    
    def extract_numbers(s: str):
        numbers = []
        current = ""
        for char in s:
            if char.isdigit():      # if it's a digit, add to current number
                current += char
            else:
                if current:         # if we have collected digits, push to list
                    numbers.append(int(current))
                    current = ""
        if current:                 # catch any trailing number
            numbers.append(int(current))
        return numbers
    
    
    def ion_mass(self, ion_name, defult_H2O = None):
        if ion_name == '???' or ion_name == None:
            return None
        
        if len(ion_name.split('/')) > 1:
            ion_name = ion_name.split('/')[0]
        
        if defult_H2O is None:
            defult_H2O = self.end_h20
            
        
        if ion_name[0] == 'y' and ion_name[:2] != 'yi':
            mass = 0
            end = int(ion_name[1:])
            for i in range(0, end):
                mass += self.rev_AA_array[i].get_mass()
            #return (mass + AA.element_masses['H2O'] + ion_charge * AA.element_masses['proton']) / ion_charge
            #return mass
            if defult_H2O is True:
                return mass + AA.element_masses['H2O'] + AA.element_masses['proton']
            else:
                return mass + AA.element_masses['proton'] + AA.element_masses[defult_H2O]
        
        elif ion_name[0] =='b' and ion_name[:2] != 'bi':
            mass = 0
            end = int(ion_name[1:])
            for i in range(0, end):
                mass += self.AA_array[i].get_mass()
            #return (mass + ion_charge * AA.element_masses['proton']) / ion_charge
            return mass + AA.element_masses['proton']
        
        elif ion_name[0] =='a' and ion_name[:2] != 'ai':
            mass = 0
            end = int(ion_name[1:])
            for i in range(0, end):
                mass += self.AA_array[i].get_mass()
            #return (mass + ion_charge * AA.element_masses['proton'] - AA.element_masses['carbonyl']) / ion_charge
            return mass - AA.element_masses['carbonyl'] + AA.element_masses['proton']
        
        
        elif ion_name[:2] == 'yi':
            the_index = Pep.extract_numbers(ion_name)
            the_index.sort()
            if len(the_index) == 2:
                start, end = the_index[0], the_index[1]
                mass = 0
                for i in range(start, end + 1):
                    mass += self.rev_AA_array[i-1].get_mass()
                #return (mass + AA.element_masses['H2O'] + ion_charge * AA.element_masses['proton']) / ion_charge
                if defult_H2O:
                    return mass + AA.element_masses['H2O']
                else:
                    return mass
            else:
                print('internal peptide number error, the index is ', the_index)
        
        elif ion_name[:2] == 'bi':
            the_index = Pep.extract_numbers(ion_name)
            the_index.sort()
            if len(the_index) == 2:
                start, end = int(the_index[0]), int(the_index[1])
                mass = 0
                for i in range(start, end + 1):
                    mass += self.AA_array[i-1].get_mass()
                #return (mass + ion_charge * AA.element_masses['proton']) / ion_charge
                return mass
            else:
                print('internal peptide number error, the index is ', the_index)
        
        elif ion_name[:2] == 'ai':
            the_index = Pep.extract_numbers(ion_name)
            the_index.sort()
            if len(the_index) == 2:
                start, end = int(the_index[0]), int(the_index[1])
                mass = 0
                for i in range(start, end + 1):
                    mass += self.AA_array[i-1].get_mass()
                #return (mass + ion_charge * AA.element_masses['proton'] - AA.element_masses['carbonyl']) / ion_charge
                return mass - AA.element_masses['carbonyl']
                
            else:
                print('internal peptide number error, the index is ', the_index)
        else:
            print('can not identify the input ion: ', ion_name)
            return None
    
    
    def ion_charge_mass(self, ion, charge):
        the_ion_mass = self.ion_mass(ion)
        return (the_ion_mass - AA.element_masses['proton'] + AA.element_masses['proton'] * charge) / charge
        
'''        
ptm_dict = {
    "Me": 14.01565, "Me2": 28.03130, "Me3": 42.04695,
    "Ac": 42.01056, "p": 79.96633, "NH2": 16.01872
}
'''


#print(Pep('[GGNFSGRMeGGFGGSR+2H]2+').AA_array)
the_pep = Pep('[EQFDDY(p)GHMRF(NH2) +3H]3+')
#print(the_pep.ion_mass('y2'))



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
        "3510": "GSHQISLDNPDYQQDFFPK",
        "test1":"YLEFISDAIIHVLHSK"}