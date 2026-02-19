*"----------------------------------------------------------------------
*"* Sample ABAP for testing
*"----------------------------------------------------------------------
CLASS zcl_sample DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS process.
  PRIVATE SECTION.
    DATA mv_value TYPE string.
ENDCLASS.

CLASS zcl_sample IMPLEMENTATION.
  METHOD process.
    DATA lv_a TYPE string.
    DATA lv_b TYPE string.
    MOVE 'hello' TO lv_a.
    ADD 1 TO mv_value.
    CONCATENATE lv_a lv_b INTO mv_value.
    CHECK lv_a IS NOT INITIAL.
    CALL METHOD me->process.
  ENDMETHOD.
ENDCLASS.
