# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: features.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'features.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0e\x66\x65\x61tures.proto*\xd6\x07\n\rFeatureEntity\x12\x12\n\x0e\x45NTITY_SPECIES\x10\x00\x12\x0f\n\x0b\x45NTITY_ITEM\x10\x01\x12\x16\n\x12\x45NTITY_ITEM_EFFECT\x10\x02\x12\x12\n\x0e\x45NTITY_ABILITY\x10\x03\x12\x11\n\rENTITY_GENDER\x10\x04\x12\x11\n\rENTITY_ACTIVE\x10\x05\x12\x12\n\x0e\x45NTITY_FAINTED\x10\x06\x12\r\n\tENTITY_HP\x10\x07\x12\x10\n\x0c\x45NTITY_MAXHP\x10\x08\x12\x11\n\rENTITY_STATUS\x10\t\x12\x16\n\x12\x45NTITY_TOXIC_TURNS\x10\n\x12\x16\n\x12\x45NTITY_SLEEP_TURNS\x10\x0b\x12\x1c\n\x18\x45NTITY_BEING_CALLED_BACK\x10\x0c\x12\x12\n\x0e\x45NTITY_TRAPPED\x10\r\x12\x19\n\x15\x45NTITY_NEWLY_SWITCHED\x10\x0e\x12\x10\n\x0c\x45NTITY_LEVEL\x10\x0f\x12\x12\n\x0e\x45NTITY_MOVEID0\x10\x10\x12\x12\n\x0e\x45NTITY_MOVEID1\x10\x11\x12\x12\n\x0e\x45NTITY_MOVEID2\x10\x12\x12\x12\n\x0e\x45NTITY_MOVEID3\x10\x13\x12\x12\n\x0e\x45NTITY_MOVEPP0\x10\x14\x12\x12\n\x0e\x45NTITY_MOVEPP1\x10\x15\x12\x12\n\x0e\x45NTITY_MOVEPP2\x10\x16\x12\x12\n\x0e\x45NTITY_MOVEPP3\x10\x17\x12\x15\n\x11\x45NTITY_HAS_STATUS\x10\x18\x12\x1a\n\x16\x45NTITY_BOOST_ATK_VALUE\x10\x19\x12\x1a\n\x16\x45NTITY_BOOST_DEF_VALUE\x10\x1a\x12\x1a\n\x16\x45NTITY_BOOST_SPA_VALUE\x10\x1b\x12\x1a\n\x16\x45NTITY_BOOST_SPD_VALUE\x10\x1c\x12\x1a\n\x16\x45NTITY_BOOST_SPE_VALUE\x10\x1d\x12\x1f\n\x1b\x45NTITY_BOOST_ACCURACY_VALUE\x10\x1e\x12\x1e\n\x1a\x45NTITY_BOOST_EVASION_VALUE\x10\x1f\x12\x15\n\x11\x45NTITY_VOLATILES0\x10 \x12\x15\n\x11\x45NTITY_VOLATILES1\x10!\x12\x15\n\x11\x45NTITY_VOLATILES2\x10\"\x12\x15\n\x11\x45NTITY_VOLATILES3\x10#\x12\x15\n\x11\x45NTITY_VOLATILES4\x10$\x12\x15\n\x11\x45NTITY_VOLATILES5\x10%\x12\x15\n\x11\x45NTITY_VOLATILES6\x10&\x12\x15\n\x11\x45NTITY_VOLATILES7\x10\'\x12\x15\n\x11\x45NTITY_VOLATILES8\x10(\x12\x0f\n\x0b\x45NTITY_SIDE\x10)\x12\x13\n\x0f\x45NTITY_HP_TOKEN\x10**(\n\x0e\x46\x65\x61tureMoveset\x12\n\n\x06MOVEID\x10\x00\x12\n\n\x06PPUSED\x10\x01*u\n\x12\x46\x65\x61tureTurnContext\x12\t\n\x05VALID\x10\x00\x12\x0e\n\nIS_MY_TURN\x10\x01\x12\n\n\x06\x41\x43TION\x10\x02\x12\x08\n\x04MOVE\x10\x03\x12\x12\n\x0eSWITCH_COUNTER\x10\x04\x12\x10\n\x0cMOVE_COUNTER\x10\x05\x12\x08\n\x04TURN\x10\x06*D\n\x0e\x46\x65\x61tureWeather\x12\x0e\n\nWEATHER_ID\x10\x00\x12\x10\n\x0cMIN_DURATION\x10\x01\x12\x10\n\x0cMAX_DURATION\x10\x02*\xfa\x04\n\x1c\x46\x65\x61tureAdditionalInformation\x12\x0f\n\x0bNUM_FAINTED\x10\x00\x12\x0c\n\x08HP_TOTAL\x10\x01\x12\x11\n\rNUM_TYPES_PAD\x10\x02\x12\x11\n\rNUM_TYPES_UNK\x10\x03\x12\x11\n\rNUM_TYPES_BUG\x10\x04\x12\x12\n\x0eNUM_TYPES_DARK\x10\x05\x12\x14\n\x10NUM_TYPES_DRAGON\x10\x06\x12\x16\n\x12NUM_TYPES_ELECTRIC\x10\x07\x12\x13\n\x0fNUM_TYPES_FAIRY\x10\x08\x12\x16\n\x12NUM_TYPES_FIGHTING\x10\t\x12\x12\n\x0eNUM_TYPES_FIRE\x10\n\x12\x14\n\x10NUM_TYPES_FLYING\x10\x0b\x12\x13\n\x0fNUM_TYPES_GHOST\x10\x0c\x12\x13\n\x0fNUM_TYPES_GRASS\x10\r\x12\x14\n\x10NUM_TYPES_GROUND\x10\x0e\x12\x11\n\rNUM_TYPES_ICE\x10\x0f\x12\x14\n\x10NUM_TYPES_NORMAL\x10\x10\x12\x14\n\x10NUM_TYPES_POISON\x10\x11\x12\x15\n\x11NUM_TYPES_PSYCHIC\x10\x12\x12\x12\n\x0eNUM_TYPES_ROCK\x10\x13\x12\x13\n\x0fNUM_TYPES_STEEL\x10\x14\x12\x15\n\x11NUM_TYPES_STELLAR\x10\x15\x12\x13\n\x0fNUM_TYPES_WATER\x10\x16\x12\x11\n\rTOTAL_POKEMON\x10\x17\x12\x0b\n\x07WISHING\x10\x18\x12\x0e\n\nMEMBER0_HP\x10\x19\x12\x0e\n\nMEMBER1_HP\x10\x1a\x12\x0e\n\nMEMBER2_HP\x10\x1b\x12\x0e\n\nMEMBER3_HP\x10\x1c\x12\x0e\n\nMEMBER4_HP\x10\x1d\x12\x0e\n\nMEMBER5_HP\x10\x1e*_\n\tEdgeTypes\x12\x12\n\x0e\x45\x44GE_TYPE_NONE\x10\x00\x12\r\n\tMOVE_EDGE\x10\x01\x12\x0f\n\x0bSWITCH_EDGE\x10\x02\x12\x0f\n\x0b\x45\x46\x46\x45\x43T_EDGE\x10\x03\x12\r\n\tCANT_EDGE\x10\x04*\xf6\x01\n\rEdgeFromTypes\x12\x12\n\x0e\x45\x44GE_FROM_NONE\x10\x00\x12\x12\n\x0e\x45\x44GE_FROM_ITEM\x10\x01\x12\x14\n\x10\x45\x44GE_FROM_EFFECT\x10\x02\x12\x12\n\x0e\x45\x44GE_FROM_MOVE\x10\x03\x12\x15\n\x11\x45\x44GE_FROM_ABILITY\x10\x04\x12\x1b\n\x17\x45\x44GE_FROM_SIDECONDITION\x10\x05\x12\x14\n\x10\x45\x44GE_FROM_STATUS\x10\x06\x12\x15\n\x11\x45\x44GE_FROM_WEATHER\x10\x07\x12\x15\n\x11\x45\x44GE_FROM_TERRAIN\x10\x08\x12\x1b\n\x17\x45\x44GE_FROM_PSEUDOWEATHER\x10\t*\xc2\x03\n\x0b\x46\x65\x61tureEdge\x12\x0f\n\x0bPOKE1_INDEX\x10\x00\x12\x0f\n\x0bPOKE2_INDEX\x10\x01\x12\x14\n\x10TURN_ORDER_VALUE\x10\x02\x12\x13\n\x0f\x45\x44GE_TYPE_TOKEN\x10\x03\x12\r\n\tMAJOR_ARG\x10\x04\x12\r\n\tMINOR_ARG\x10\x05\x12\x0e\n\nMOVE_TOKEN\x10\x06\x12\x0e\n\nITEM_TOKEN\x10\x07\x12\x11\n\rABILITY_TOKEN\x10\x08\x12\x13\n\x0f\x46ROM_TYPE_TOKEN\x10\t\x12\x15\n\x11\x46ROM_SOURCE_TOKEN\x10\n\x12\x10\n\x0c\x44\x41MAGE_TOKEN\x10\x0b\x12\x10\n\x0c\x45\x46\x46\x45\x43T_TOKEN\x10\x0c\x12\x13\n\x0f\x42OOST_ATK_VALUE\x10\r\x12\x13\n\x0f\x42OOST_DEF_VALUE\x10\x0e\x12\x13\n\x0f\x42OOST_SPA_VALUE\x10\x0f\x12\x13\n\x0f\x42OOST_SPD_VALUE\x10\x10\x12\x13\n\x0f\x42OOST_SPE_VALUE\x10\x11\x12\x18\n\x14\x42OOST_ACCURACY_VALUE\x10\x12\x12\x17\n\x13\x42OOST_EVASION_VALUE\x10\x13\x12\x10\n\x0cSTATUS_TOKEN\x10\x14\x12\x17\n\x13\x45\x44GE_AFFECTING_SIDE\x10\x15\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'features_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_FEATUREENTITY']._serialized_start=19
  _globals['_FEATUREENTITY']._serialized_end=1001
  _globals['_FEATUREMOVESET']._serialized_start=1003
  _globals['_FEATUREMOVESET']._serialized_end=1043
  _globals['_FEATURETURNCONTEXT']._serialized_start=1045
  _globals['_FEATURETURNCONTEXT']._serialized_end=1162
  _globals['_FEATUREWEATHER']._serialized_start=1164
  _globals['_FEATUREWEATHER']._serialized_end=1232
  _globals['_FEATUREADDITIONALINFORMATION']._serialized_start=1235
  _globals['_FEATUREADDITIONALINFORMATION']._serialized_end=1869
  _globals['_EDGETYPES']._serialized_start=1871
  _globals['_EDGETYPES']._serialized_end=1966
  _globals['_EDGEFROMTYPES']._serialized_start=1969
  _globals['_EDGEFROMTYPES']._serialized_end=2215
  _globals['_FEATUREEDGE']._serialized_start=2218
  _globals['_FEATUREEDGE']._serialized_end=2668
# @@protoc_insertion_point(module_scope)
