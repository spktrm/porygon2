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




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0e\x66\x65\x61tures.proto*\x86\x08\n\rFeatureEntity\x12\x12\n\x0e\x45NTITY_SPECIES\x10\x00\x12\x0f\n\x0b\x45NTITY_ITEM\x10\x01\x12\x16\n\x12\x45NTITY_ITEM_EFFECT\x10\x02\x12\x12\n\x0e\x45NTITY_ABILITY\x10\x03\x12\x11\n\rENTITY_GENDER\x10\x04\x12\x11\n\rENTITY_ACTIVE\x10\x05\x12\x12\n\x0e\x45NTITY_FAINTED\x10\x06\x12\r\n\tENTITY_HP\x10\x07\x12\x10\n\x0c\x45NTITY_MAXHP\x10\x08\x12\x13\n\x0f\x45NTITY_HP_RATIO\x10\t\x12\x11\n\rENTITY_STATUS\x10\n\x12\x16\n\x12\x45NTITY_TOXIC_TURNS\x10\x0b\x12\x16\n\x12\x45NTITY_SLEEP_TURNS\x10\x0c\x12\x1c\n\x18\x45NTITY_BEING_CALLED_BACK\x10\r\x12\x12\n\x0e\x45NTITY_TRAPPED\x10\x0e\x12\x19\n\x15\x45NTITY_NEWLY_SWITCHED\x10\x0f\x12\x10\n\x0c\x45NTITY_LEVEL\x10\x10\x12\x12\n\x0e\x45NTITY_MOVEID0\x10\x11\x12\x12\n\x0e\x45NTITY_MOVEID1\x10\x12\x12\x12\n\x0e\x45NTITY_MOVEID2\x10\x13\x12\x12\n\x0e\x45NTITY_MOVEID3\x10\x14\x12\x12\n\x0e\x45NTITY_MOVEPP0\x10\x15\x12\x12\n\x0e\x45NTITY_MOVEPP1\x10\x16\x12\x12\n\x0e\x45NTITY_MOVEPP2\x10\x17\x12\x12\n\x0e\x45NTITY_MOVEPP3\x10\x18\x12\x15\n\x11\x45NTITY_HAS_STATUS\x10\x19\x12\x1a\n\x16\x45NTITY_BOOST_ATK_VALUE\x10\x1a\x12\x1a\n\x16\x45NTITY_BOOST_DEF_VALUE\x10\x1b\x12\x1a\n\x16\x45NTITY_BOOST_SPA_VALUE\x10\x1c\x12\x1a\n\x16\x45NTITY_BOOST_SPD_VALUE\x10\x1d\x12\x1a\n\x16\x45NTITY_BOOST_SPE_VALUE\x10\x1e\x12\x1f\n\x1b\x45NTITY_BOOST_ACCURACY_VALUE\x10\x1f\x12\x1e\n\x1a\x45NTITY_BOOST_EVASION_VALUE\x10 \x12\x15\n\x11\x45NTITY_VOLATILES0\x10!\x12\x15\n\x11\x45NTITY_VOLATILES1\x10\"\x12\x15\n\x11\x45NTITY_VOLATILES2\x10#\x12\x15\n\x11\x45NTITY_VOLATILES3\x10$\x12\x15\n\x11\x45NTITY_VOLATILES4\x10%\x12\x15\n\x11\x45NTITY_VOLATILES5\x10&\x12\x15\n\x11\x45NTITY_VOLATILES6\x10\'\x12\x15\n\x11\x45NTITY_VOLATILES7\x10(\x12\x15\n\x11\x45NTITY_VOLATILES8\x10)\x12\x0f\n\x0b\x45NTITY_SIDE\x10*\x12\x16\n\x12\x45NTITY_TYPECHANGE0\x10+\x12\x16\n\x12\x45NTITY_TYPECHANGE1\x10,*Q\n\x11MovesetActionType\x12\x1c\n\x18MOVESET_ACTION_TYPE_MOVE\x10\x00\x12\x1e\n\x1aMOVESET_ACTION_TYPE_SWITCH\x10\x01*\xbe\x01\n\x0e\x46\x65\x61tureMoveset\x12\x15\n\x11MOVESET_ACTION_ID\x10\x00\x12\x12\n\x0eMOVESET_PPUSED\x10\x01\x12\x11\n\rMOVESET_LEGAL\x10\x02\x12\x10\n\x0cMOVESET_SIDE\x10\x03\x12\x17\n\x13MOVESET_ACTION_TYPE\x10\x04\x12\x16\n\x12MOVESET_EST_DAMAGE\x10\x05\x12\x13\n\x0fMOVESET_MOVE_ID\x10\x06\x12\x16\n\x12MOVESET_SPECIES_ID\x10\x07*t\n\tEdgeTypes\x12\x12\n\x0e\x45\x44GE_TYPE_NONE\x10\x00\x12\r\n\tMOVE_EDGE\x10\x01\x12\x0f\n\x0bSWITCH_EDGE\x10\x02\x12\x0f\n\x0b\x45\x46\x46\x45\x43T_EDGE\x10\x03\x12\r\n\tCANT_EDGE\x10\x04\x12\x13\n\x0f\x45\x44GE_TYPE_START\x10\x05*\xf6\x01\n\rEdgeFromTypes\x12\x12\n\x0e\x45\x44GE_FROM_NONE\x10\x00\x12\x12\n\x0e\x45\x44GE_FROM_ITEM\x10\x01\x12\x14\n\x10\x45\x44GE_FROM_EFFECT\x10\x02\x12\x12\n\x0e\x45\x44GE_FROM_MOVE\x10\x03\x12\x15\n\x11\x45\x44GE_FROM_ABILITY\x10\x04\x12\x1b\n\x17\x45\x44GE_FROM_SIDECONDITION\x10\x05\x12\x14\n\x10\x45\x44GE_FROM_STATUS\x10\x06\x12\x15\n\x11\x45\x44GE_FROM_WEATHER\x10\x07\x12\x15\n\x11\x45\x44GE_FROM_TERRAIN\x10\x08\x12\x1b\n\x17\x45\x44GE_FROM_PSEUDOWEATHER\x10\t*\x9d\x03\n\x13\x46\x65\x61tureAbsoluteEdge\x12\x19\n\x15\x45\x44GE_TURN_ORDER_VALUE\x10\x00\x12\x13\n\x0f\x45\x44GE_TYPE_TOKEN\x10\x01\x12\x13\n\x0f\x45\x44GE_WEATHER_ID\x10\x02\x12\x1d\n\x19\x45\x44GE_WEATHER_MIN_DURATION\x10\x03\x12\x1d\n\x19\x45\x44GE_WEATHER_MAX_DURATION\x10\x04\x12\x13\n\x0f\x45\x44GE_TERRAIN_ID\x10\x05\x12\x1d\n\x19\x45\x44GE_TERRAIN_MIN_DURATION\x10\x06\x12\x1d\n\x19\x45\x44GE_TERRAIN_MAX_DURATION\x10\x07\x12\x19\n\x15\x45\x44GE_PSEUDOWEATHER_ID\x10\x08\x12#\n\x1f\x45\x44GE_PSEUDOWEATHER_MIN_DURATION\x10\t\x12#\n\x1f\x45\x44GE_PSEUDOWEATHER_MAX_DURATION\x10\n\x12\x16\n\x12\x45\x44GE_REQUEST_COUNT\x10\x0b\x12\x0e\n\nEDGE_VALID\x10\x0c\x12\x0e\n\nEDGE_INDEX\x10\r\x12\x13\n\x0f\x45\x44GE_TURN_VALUE\x10\x0e*\xec\x04\n\x13\x46\x65\x61tureRelativeEdge\x12\x12\n\x0e\x45\x44GE_MAJOR_ARG\x10\x00\x12\x13\n\x0f\x45\x44GE_MINOR_ARG0\x10\x01\x12\x13\n\x0f\x45\x44GE_MINOR_ARG1\x10\x02\x12\x13\n\x0f\x45\x44GE_MINOR_ARG2\x10\x03\x12\x13\n\x0f\x45\x44GE_MINOR_ARG3\x10\x04\x12\x15\n\x11\x45\x44GE_ACTION_TOKEN\x10\x05\x12\x13\n\x0f\x45\x44GE_ITEM_TOKEN\x10\x06\x12\x16\n\x12\x45\x44GE_ABILITY_TOKEN\x10\x07\x12\x18\n\x14\x45\x44GE_FROM_TYPE_TOKEN\x10\x08\x12\x1a\n\x16\x45\x44GE_FROM_SOURCE_TOKEN\x10\t\x12\x15\n\x11\x45\x44GE_DAMAGE_RATIO\x10\n\x12\x13\n\x0f\x45\x44GE_HEAL_RATIO\x10\x0b\x12\x15\n\x11\x45\x44GE_EFFECT_TOKEN\x10\x0c\x12\x18\n\x14\x45\x44GE_BOOST_ATK_VALUE\x10\r\x12\x18\n\x14\x45\x44GE_BOOST_DEF_VALUE\x10\x0e\x12\x18\n\x14\x45\x44GE_BOOST_SPA_VALUE\x10\x0f\x12\x18\n\x14\x45\x44GE_BOOST_SPD_VALUE\x10\x10\x12\x18\n\x14\x45\x44GE_BOOST_SPE_VALUE\x10\x11\x12\x1d\n\x19\x45\x44GE_BOOST_ACCURACY_VALUE\x10\x12\x12\x1c\n\x18\x45\x44GE_BOOST_EVASION_VALUE\x10\x13\x12\x15\n\x11\x45\x44GE_STATUS_TOKEN\x10\x14\x12\x18\n\x14\x45\x44GE_SIDECONDITIONS0\x10\x15\x12\x18\n\x14\x45\x44GE_SIDECONDITIONS1\x10\x16\x12\x15\n\x11\x45\x44GE_TOXIC_SPIKES\x10\x17\x12\x0f\n\x0b\x45\x44GE_SPIKES\x10\x18\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'features_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_FEATUREENTITY']._serialized_start=19
  _globals['_FEATUREENTITY']._serialized_end=1049
  _globals['_MOVESETACTIONTYPE']._serialized_start=1051
  _globals['_MOVESETACTIONTYPE']._serialized_end=1132
  _globals['_FEATUREMOVESET']._serialized_start=1135
  _globals['_FEATUREMOVESET']._serialized_end=1325
  _globals['_EDGETYPES']._serialized_start=1327
  _globals['_EDGETYPES']._serialized_end=1443
  _globals['_EDGEFROMTYPES']._serialized_start=1446
  _globals['_EDGEFROMTYPES']._serialized_end=1692
  _globals['_FEATUREABSOLUTEEDGE']._serialized_start=1695
  _globals['_FEATUREABSOLUTEEDGE']._serialized_end=2108
  _globals['_FEATURERELATIVEEDGE']._serialized_start=2111
  _globals['_FEATURERELATIVEEDGE']._serialized_end=2731
# @@protoc_insertion_point(module_scope)
