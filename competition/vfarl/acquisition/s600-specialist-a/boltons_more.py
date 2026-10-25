"""Explicit source-reviewed Boltons representation, span and traversal plans.

Empty publisher descriptions are retained and disclosed; the minimal curated
behavior statements below are used in the actual replay task from the outset.
No replay is claimed under an empty description and no verifier is weakened.
"""
import argparse
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
PLANS=[
 ('0fcds2q4','S1','infer_positional_format_args','formatutils','anonymous_format_field_numbering_and_literal_tail',"from boltons.formatutils import infer_positional_format_args\nr=infer_positional_format_args('{} {:d} tail');print('explicit_fields',r)\nassert r=='{0} {1:d} tail'"),
 ('9pxz5n3d','S1','BaseFormatField.set_conv','formatutils','format_conversion_representation_and_unset_converter',"from boltons.formatutils import BaseFormatField\nf=BaseFormatField('x',conv='s');print('conversion',f.conv,f.conv_func)\nassert f.conv=='s' and f.conv_func is None\nf.set_conv('r');assert f.conv=='r' and f.conv_func is None"),
 ('dj6naujo','S1','asciify','strutils','unicode_deaccenting_to_ascii_bytes',"from boltons.strutils import asciify\nr=asciify('Beyoncé');print('ascii_bytes',r)\nassert r==b'Beyonce'\nassert asciify('abc')==b'abc'"),
 ('0suuttt3','S1','parse_int_list','strutils','inclusive_range_grammar_and_sorted_integer_decoding',"from boltons.strutils import parse_int_list\nr=parse_int_list('5-3,1,8');print('decoded_ranges',r)\nassert r==[1,3,4,5,8]"),
 ('n7a4fy5x','S1','format_int_list','strutils','canonical_contiguous_range_serialization',"from boltons.strutils import format_int_list\nr=format_int_list([1,2,4,5,6,8]);print('encoded_ranges',r)\nassert r=='1-2,4-6,8'"),
 ('iujey63s','S1','gzip_bytes','strutils','complete_gzip_payload_preserves_uncompressed_bytes',"from boltons.strutils import gzip_bytes\nimport gzip\npayload=b'abc'*10;r=gzip_bytes(payload);print('compressed_bytes',len(r))\nassert gzip.decompress(r)==payload"),
 ('wkfvmv7j','S1','strip_ansi','strutils','ansi_escape_removal_preserves_byte_representation',"from boltons.strutils import strip_ansi\nr=strip_ansi(b'\x1b[31mred\x1b[0m');print('cleaned_bytes',r)\nassert r==b'red'\nassert strip_ansi('\x1b[31mred\x1b[0m')=='red'"),
 ('dy94l0zg','S1','MultiReplace.__init__','strutils','literal_or_regex_pattern_compilation_for_parallel_replace',"from boltons.strutils import MultiReplace\nr=MultiReplace({'a.':'Z'}).sub('aX a.');print('literal_replace',r)\nassert r=='aX Z'\nassert MultiReplace([('a.','Z')],regex=True).sub('aX a.')=='Z Z'"),
 ('zwmnsgsz','S1','indent','strutils','line_prefix_and_newline_representation_preservation',"from boltons.strutils import indent\nr=indent('a'+chr(10)+chr(10)+'b','> ');print('indented',repr(r))\nassert r=='> a'+chr(10)+chr(10)+'> b'"),
 ('1iqe9f42','S2','iter_splitlines','strutils','line_ending_spans_and_terminal_empty_line',"from boltons.strutils import iter_splitlines\nr=list(iter_splitlines('a'+chr(13)+chr(10)+'b'+chr(10)));print('lines',r)\nassert r==['a','b','']"),
 ('elfsbsty','S2','chunk_ranges','iterutils','offset_overlap_and_aligned_chunk_spans',"from boltons.iterutils import chunk_ranges\nr=list(chunk_ranges(input_offset=10,input_size=10,chunk_size=5,overlap_size=1));print('ranges',r)\nassert r==[(10,15),(14,19),(18,20)]\nassert list(chunk_ranges(input_offset=4,input_size=15,chunk_size=5,align=True))==[(4,5),(5,10),(10,15),(15,19)]"),
 ('tsm3kl3m','S2','chunked_iter','iterutils','fixed_chunk_span_and_optional_tail_padding',"from boltons.iterutils import chunked_iter\nr=list(chunked_iter(range(7),3));print('chunks',r)\nassert r==[[0,1,2],[3,4,5],[6]]\nassert list(chunked_iter(range(7),3,fill=None))==[[0,1,2],[3,4,5],[6,None,None]]"),
 ('7o1bci14','S2','windowed_iter','iterutils','sliding_window_offsets_width_and_fill',"from boltons.iterutils import windowed_iter\nr=list(windowed_iter([0,1,2],2));print('windows',r)\nassert r==[(0,1),(1,2)]\nassert list(windowed_iter([0,1,2],2,fill=9))==[(0,1),(1,2),(2,9)]"),
 ('r93zx4tx','S3','first','iterutils','first_matching_item_selection_with_default',"from boltons.iterutils import first\nr=first([0,'',3,4]);print('first_match',r)\nassert r==3\nassert first([1,3,4],key=lambda x:x%2==0)==4\nassert first([],default=7)==7"),
 ('l2duaj4k','S4','rstrip_iter','iterutils','pending_strip_run_preserved_until_known_to_be_tail',"from boltons.iterutils import rstrip_iter\nr=list(rstrip_iter([1,None,2,None,None]));print('right_trimmed',r)\nassert r==[1,None,2]"),
 ('3e1omssu','S4','backoff_iter','iterutils','bounded_geometric_progression_and_repeat_plateau',"from boltons.iterutils import backoff_iter\nr=list(backoff_iter(1,10,count=5,jitter=False));print('backoff_curve',r)\nassert r==[1,2,4,8,10]\nassert list(backoff_iter(.25,100,factor=10,jitter=False))==[.25,2.5,25,100]"),
 ('ws1wkds5','S5','remap','iterutils','recursive_heterogeneous_tree_transform_and_visit_filter',"from boltons.iterutils import remap\nroot={'a':[None,1],'b':2};r=remap(root,lambda p,k,v:v is not None);print('remapped_tree',r)\nassert r=={'a':[1],'b':2}\nassert root=={'a':[None,1],'b':2}"),
 ('jbxc74e1','S5','research','iterutils','recursive_structural_query_reports_complete_paths',"from boltons.iterutils import research\nroot={'a':{'b':1},'c':[2,'z']};r=research(root,query=lambda p,k,v:isinstance(v,int));print('nested_matches',r)\nassert sorted(r)==[(('a','b'),1),(('c',0),2)]"),
 ('obaisyei','S5','get_path','iterutils','heterogeneous_nested_path_lookup_and_missing_default',"from boltons.iterutils import get_path\nroot={'a':[{'b':3}]};r=get_path(root,('a',0,'b'));print('nested_value',r)\nassert r==3\nmarker=object();assert get_path(root,('a',2),default=marker) is marker"),
]
PROMPTS={
 '0fcds2q4':"Fix boltons.formatutils.infer_positional_format_args. Anonymous format fields should receive consecutive explicit argument numbers while escaped braces, formatting details, and trailing literal text are preserved.\n",
 '9pxz5n3d':"Fix conversion handling in boltons.formatutils.BaseFormatField. Constructing or updating a format field should preserve its declared conversion value, including the default with no conversion, and leave the converter placeholder in its documented unset state.\n",
 'dj6naujo':"Fix boltons.strutils.asciify. It should produce ASCII bytes, deaccent supported Unicode characters, and apply its ignore option to remaining unrepresentable characters while preserving ordinary ASCII input.\n",
 '0suuttt3':"Fix boltons.strutils.parse_int_list. Decode delimited integers and inclusive ranges into a sorted integer list, including ranges whose endpoints are written in descending order and inputs with ordinary individual numbers. Preserve the public delimiter options.\n",
 'n7a4fy5x':"Fix boltons.strutils.format_int_list. Serialize positive integers as a sorted range string, combining contiguous runs and retaining separate gaps and isolated values. Preserve the documented delimiter and spacing options.\n",
 'iujey63s':"Fix boltons.strutils.gzip_bytes. Compressing a byte payload should return a complete valid gzip byte stream that decompresses to the original payload in the original order. Preserve the requested compression level and normal empty-input behavior.\n",
 'wkfvmv7j':"Fix boltons.strutils.strip_ansi. Remove ANSI escape sequences from text and byte inputs without changing the remaining content or the input's text, bytes, or bytearray representation.\n",
 'dy94l0zg':"Fix construction of boltons.strutils.MultiReplace. Search strings should be treated literally by default, with an explicit option for regular expressions and their flags. Substitutions should use the supplied replacement mapping in a single pass.\n",
 'zwmnsgsz':"Fix boltons.strutils.indent. The margin should prefix each selected line, with line breaks preserved through the requested newline separator. Keep the documented default behavior for blank lines and custom line-selection predicates.\n",
 '1iqe9f42':"Fix boltons.strutils.iter_splitlines. Yield the original line contents at the supported line-ending boundaries, including CRLF and a final empty line when text ends with a line ending. Avoid dropping characters or including the line-ending characters in a line.\n",
 'elfsbsty':"Fix boltons.iterutils.chunk_ranges. Return start/end spans with the requested chunk size, input offset, overlap, and alignment. Spans should cover the requested input interval and end at its boundary without skipping overlapping positions.\n",
 'tsm3kl3m':"Fix boltons.iterutils.chunked_iter. Produce consecutive chunks of the requested size in input order. A final short chunk should remain short unless fill is explicitly supplied, including fill=None. Preserve supported sequence and byte inputs.\n",
 '7o1bci14':"Fix boltons.iterutils.windowed_iter. Return sliding windows of exactly the requested width beginning at consecutive source positions. Without fill, omit incomplete windows; with fill, preserve the exact fill value and include trailing windows through the last source item.\n",
 'r93zx4tx':"Fix boltons.iterutils.first. Return the first truthy item, or the first item matching the supplied key predicate. If no item matches, return the supplied default without reversing the predicate.\n",
 'l2duaj4k':"Fix boltons.iterutils.rstrip_iter. Remove only the final consecutive run of the requested strip value. Equal values followed by another ordinary item belong to the output and must retain their original order.\n",
 '3e1omssu':"Fix boltons.iterutils.backoff_iter. Generate the documented geometric sequence from start toward stop, bounded at stop, with the requested factor, count, and repeat behavior. Preserve the documented optional jitter behavior.\n",
 'ws1wkds5':"Fix boltons.iterutils.remap. Recursively transform heterogeneous nested containers using the enter, visit, and exit callbacks. Keep or drop values according to the visitor, preserve the documented handling of repeated and self-referential containers, and honor the existing exception and trace options.\n",
 'jbxc74e1':"Fix boltons.iterutils.research. Recursively inspect nested containers and return each matching value together with its complete path. Preserve the query predicate, optional traversal callback, and documented exception behavior.\n",
}


def helper():
    spec=importlib.util.spec_from_file_location('s600a_more_data',HERE/'collect.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'));parser.add_argument('--start',type=int,default=0);parser.add_argument('--limit',type=int);parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    if args.stage=='prepare':helper().prepare_boltons(PLANS,'reviewed-boltons-more-candidates.json',PROMPTS)
    else:helper().run(args.start,args.limit,'boltons_more',args.attempt)
