// source: features.proto
/**
 * @fileoverview
 * @enhanceable
 * @suppress {missingRequire} reports error on implicit type usages.
 * @suppress {messageConventions} JS Compiler reports an error if a variable or
 *     field starts with 'MSG_' and isn't a translatable message.
 * @public
 */
// GENERATED CODE -- DO NOT EDIT!
/* eslint-disable */
// @ts-nocheck

var jspb = require('google-protobuf');
var goog = jspb;
var global = (function() { return this || window || global || self || Function('return this')(); }).call(null);

goog.exportSymbol('proto.EdgeFromTypes', null, global);
goog.exportSymbol('proto.EdgeTypes', null, global);
goog.exportSymbol('proto.FeatureAdditionalInformation', null, global);
goog.exportSymbol('proto.FeatureEdge', null, global);
goog.exportSymbol('proto.FeatureEntity', null, global);
goog.exportSymbol('proto.FeatureMoveset', null, global);
goog.exportSymbol('proto.FeatureTurnContext', null, global);
goog.exportSymbol('proto.FeatureWeather', null, global);
/**
 * @enum {number}
 */
proto.FeatureEntity = {
  ENTITY_SPECIES: 0,
  ENTITY_ITEM: 1,
  ENTITY_ITEM_EFFECT: 2,
  ENTITY_ABILITY: 3,
  ENTITY_GENDER: 4,
  ENTITY_ACTIVE: 5,
  ENTITY_FAINTED: 6,
  ENTITY_HP: 7,
  ENTITY_MAXHP: 8,
  ENTITY_STATUS: 9,
  ENTITY_TOXIC_TURNS: 10,
  ENTITY_SLEEP_TURNS: 11,
  ENTITY_BEING_CALLED_BACK: 12,
  ENTITY_TRAPPED: 13,
  ENTITY_NEWLY_SWITCHED: 14,
  ENTITY_LEVEL: 15,
  ENTITY_MOVEID0: 16,
  ENTITY_MOVEID1: 17,
  ENTITY_MOVEID2: 18,
  ENTITY_MOVEID3: 19,
  ENTITY_MOVEPP0: 20,
  ENTITY_MOVEPP1: 21,
  ENTITY_MOVEPP2: 22,
  ENTITY_MOVEPP3: 23,
  ENTITY_HAS_STATUS: 24,
  ENTITY_BOOST_ATK_VALUE: 25,
  ENTITY_BOOST_DEF_VALUE: 26,
  ENTITY_BOOST_SPA_VALUE: 27,
  ENTITY_BOOST_SPD_VALUE: 28,
  ENTITY_BOOST_SPE_VALUE: 29,
  ENTITY_BOOST_ACCURACY_VALUE: 30,
  ENTITY_BOOST_EVASION_VALUE: 31,
  ENTITY_VOLATILES0: 32,
  ENTITY_VOLATILES1: 33,
  ENTITY_VOLATILES2: 34,
  ENTITY_VOLATILES3: 35,
  ENTITY_VOLATILES4: 36,
  ENTITY_VOLATILES5: 37,
  ENTITY_VOLATILES6: 38,
  ENTITY_VOLATILES7: 39,
  ENTITY_VOLATILES8: 40,
  ENTITY_SIDE: 41,
  ENTITY_HP_TOKEN: 42
};

/**
 * @enum {number}
 */
proto.FeatureMoveset = {
  MOVEID: 0,
  PPUSED: 1
};

/**
 * @enum {number}
 */
proto.FeatureTurnContext = {
  VALID: 0,
  IS_MY_TURN: 1,
  ACTION: 2,
  MOVE: 3,
  SWITCH_COUNTER: 4,
  MOVE_COUNTER: 5,
  TURN: 6
};

/**
 * @enum {number}
 */
proto.FeatureWeather = {
  WEATHER_ID: 0,
  MIN_DURATION: 1,
  MAX_DURATION: 2
};

/**
 * @enum {number}
 */
proto.FeatureAdditionalInformation = {
  NUM_FAINTED: 0,
  HP_TOTAL: 1,
  NUM_TYPES_PAD: 2,
  NUM_TYPES_UNK: 3,
  NUM_TYPES_BUG: 4,
  NUM_TYPES_DARK: 5,
  NUM_TYPES_DRAGON: 6,
  NUM_TYPES_ELECTRIC: 7,
  NUM_TYPES_FAIRY: 8,
  NUM_TYPES_FIGHTING: 9,
  NUM_TYPES_FIRE: 10,
  NUM_TYPES_FLYING: 11,
  NUM_TYPES_GHOST: 12,
  NUM_TYPES_GRASS: 13,
  NUM_TYPES_GROUND: 14,
  NUM_TYPES_ICE: 15,
  NUM_TYPES_NORMAL: 16,
  NUM_TYPES_POISON: 17,
  NUM_TYPES_PSYCHIC: 18,
  NUM_TYPES_ROCK: 19,
  NUM_TYPES_STEEL: 20,
  NUM_TYPES_STELLAR: 21,
  NUM_TYPES_WATER: 22,
  TOTAL_POKEMON: 23,
  WISHING: 24,
  MEMBER0_HP: 25,
  MEMBER1_HP: 26,
  MEMBER2_HP: 27,
  MEMBER3_HP: 28,
  MEMBER4_HP: 29,
  MEMBER5_HP: 30
};

/**
 * @enum {number}
 */
proto.EdgeTypes = {
  EDGE_TYPE_NONE: 0,
  MOVE_EDGE: 1,
  SWITCH_EDGE: 2,
  EFFECT_EDGE: 3,
  CANT_EDGE: 4
};

/**
 * @enum {number}
 */
proto.EdgeFromTypes = {
  EDGE_FROM_NONE: 0,
  EDGE_FROM_ITEM: 1,
  EDGE_FROM_EFFECT: 2,
  EDGE_FROM_MOVE: 3,
  EDGE_FROM_ABILITY: 4,
  EDGE_FROM_SIDECONDITION: 5,
  EDGE_FROM_STATUS: 6,
  EDGE_FROM_WEATHER: 7,
  EDGE_FROM_TERRAIN: 8,
  EDGE_FROM_PSEUDOWEATHER: 9
};

/**
 * @enum {number}
 */
proto.FeatureEdge = {
  POKE1_INDEX: 0,
  POKE2_INDEX: 1,
  TURN_ORDER_VALUE: 2,
  EDGE_TYPE_TOKEN: 3,
  MAJOR_ARG: 4,
  MINOR_ARG: 5,
  MOVE_TOKEN: 6,
  ITEM_TOKEN: 7,
  ABILITY_TOKEN: 8,
  FROM_TYPE_TOKEN: 9,
  FROM_SOURCE_TOKEN: 10,
  DAMAGE_TOKEN: 11,
  EFFECT_TOKEN: 12,
  BOOST_ATK_VALUE: 13,
  BOOST_DEF_VALUE: 14,
  BOOST_SPA_VALUE: 15,
  BOOST_SPD_VALUE: 16,
  BOOST_SPE_VALUE: 17,
  BOOST_ACCURACY_VALUE: 18,
  BOOST_EVASION_VALUE: 19,
  STATUS_TOKEN: 20,
  EDGE_AFFECTING_SIDE: 21
};

goog.object.extend(exports, proto);
