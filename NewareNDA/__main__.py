'''
Script to convert Neware NDA files to other file formats. The default
output format is csv. Other formats may require additional packages to be installed.
'''
import argparse
from logging import _nameToLevel
import pandas as pd
import NewareNDA


def main():
    output_cmd = {
        'csv': lambda df, f: pd.DataFrame.to_csv(df, f, index=False),
        'excel': lambda df, f: pd.DataFrame.to_excel(df, f, index=False),
        'feather': pd.DataFrame.to_feather,
        'hdf': lambda df, f: pd.DataFrame.to_hdf(df, f, key='Index'),
        'json': pd.DataFrame.to_json,
        'parquet': pd.DataFrame.to_parquet,
        'pickle': pd.DataFrame.to_pickle,
        'stata': pd.DataFrame.to_stata
    }

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_file', help='Input file')
    parser.add_argument('out_file', help='Output file')
    parser.add_argument('-f', '--format', default='csv',
                        choices=output_cmd.keys())
    parser.add_argument('-s', '--no_software_cycle_number', action='store_false',
                        help='Generate cycle number field to match older versions of BTSDA.')
    parser.add_argument('-v', '--version', help='Show version',
                        action='version', version=NewareNDA.__version__)
    parser.add_argument('-l', '--log_level', choices=list(_nameToLevel.keys()), default='INFO',
                        help='Set the log level for NewareNDA')
    parser.add_argument('-c', '--cycle_mode', choices=['chg', 'dchg', 'auto'], default='chg',
                        help='Select cycle increment mode.')
    args = parser.parse_args()

    df = NewareNDA.read(args.in_file, args.no_software_cycle_number, cycle_mode=args.cycle_mode, log_level=args.log_level)
    output_cmd[args.format](df, args.out_file)


if __name__ == '__main__':
    main()
